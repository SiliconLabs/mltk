from typing import Union
import logging
import os
import json
import pprint

from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight


from mltk.utils.python import contains_class_type, prepend_exception_msg
from mltk.utils.path import clean_directory
from mltk.utils import gpu

from .model import (
    MltkModel, 
    KerasModel, 
    DatasetMixin,
    TrainMixin,
    load_mltk_model,
)
from .utils import get_mltk_logger
from .summarize_model import summarize_model
from .quantize_model import quantize_model
from .training_results import TrainingResults





def train_model(
    model: Union[MltkModel, str],
    weights: str = None,
    epochs: int = None,
    resume_epoch: int=0,
    verbose: bool=None,
    clean:bool=False,
    quantize:bool=True,
    create_archive:bool=True,
    show:bool=False,
    test:bool=False,
) -> TrainingResults:
    """Train a model using Keras and Tensorflow
    
    .. seealso::
       * `Model Training Guide <https://siliconlabs.github.io/mltk/docs/guides/model_training.html>`_
       * `Model Training API Examples <https://siliconlabs.github.io/mltk/mltk/examples/train_model.html>`_
       * `KerasModel.fit() <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_

    Args:
        model: :py:class:`mltk.core.MltkModel` instance, name of MLTK model, path to model specification script(.py)
            __Note:__ If the model is in "test mode" then the model will train for 1 epoch
        weights: Optional file path of model weights to load before training
        epochs: Optional, number of epochs to train model. This overrides the mltk_model.epochs attribute
        resume_epoch: Optional, resuming training at the given epoch
        verbose: Optional, Verbosely print to logger while training 
        clean: Optional, Clean the log directory before training
        quantize: Optional, quantize the model after training successfully completes
        create_archive: Optional, create an archive (.mltk.zip) of the training results and generated model files
        show: Optional, show the training results diagram
        test: Optional, load the model in "test mode" if true.

    Returns:
        The model TrainingResults
    """
    if isinstance(model, MltkModel):
        mltk_model = model
        if test:
            mltk_model.enable_test_mode()

    elif isinstance(model, str):
        if model.endswith(('.tflite', '.h5', '.zip')):
            raise ValueError(
                'Must provide name of MLTK model '
                'or path model specification script(.py)'
            )
        mltk_model = load_mltk_model(model, test=test)
    else:
        raise ValueError(
            'Must provide MltkModel instance, name of MLTK model, or path to '
            'model specification script(.py)'
        )

    if not isinstance(mltk_model, TrainMixin):
        raise ValueError('model argument must be an MltkModel instance that inherits TrainMixin')
    if not isinstance(mltk_model, DatasetMixin):
        raise ValueError('model argument must be an MltkModel instance that inherits DatasetMixin')

    logger = get_mltk_logger()

    # Ensure the MltkModel archive is writable before we start training
    if create_archive:
        mltk_model.check_archive_file_is_writable(throw_exception=True)

    # Clean the log directory if necessary
    if clean is not None:
        _clear_log_directory(mltk_model, logger=logger, recursive=clean)

    # Create the training logger
    logger = mltk_model.create_logger('train', parent=logger)
    gpu.initialize(logger=logger)

    try:
        mltk_model.load_dataset(subset='training', logger=logger, test=mltk_model.test_mode_enabled)
    except Exception as e:
        prepend_exception_msg(e, 'Failed to load model training dataset')
        raise

    # Build the MLTK model's corresponding Keras model
    try:
        keras_model = mltk_model.build_model_function(mltk_model)
    except Exception as e:
        prepend_exception_msg(e, 'Failed to build Keras model')
        raise 

    # Load the weights into the model if necessary
    try:
        if weights:
            weights_path = mltk_model.get_weights_path(weights)
            logger.info(f'Loading weights file: {weights_path}')
            keras_model.load_weights(weights_path)
    except Exception as e:
        prepend_exception_msg(e, 'Failed to load weights into Keras model')
        raise 

    # Generate a summary of the model
    try:
        summary = summarize_model( 
            mltk_model,
            built_model=keras_model
        )
        logger.info(summary)
        with open(f'{mltk_model.log_dir}/{mltk_model.name}.h5.summary.txt', 'w') as f:
            f.write(summary)
    except Exception as e:
        logger.debug(f'Failed to generate model summary, err: {e}', exc_info=e)


    logger.info(mltk_model.summarize_dataset())

    epochs = epochs or mltk_model.epochs
    callbacks = _get_keras_callbacks(mltk_model, epochs=epochs, logger=logger)
    epochs = _get_epochs(mltk_model, epochs=epochs, callbacks=callbacks, logger=logger)
    initial_epoch = _try_resume_training(mltk_model, keras_model=keras_model, epochs=epochs, resume_epoch=resume_epoch, logger=logger)
    try:
        class_weights = compute_class_weights(mltk_model, logger=logger)
    except Exception as e:
        class_weights = None
        logger.warning("Failed to compute class weights\nSet my_model.class_weights = 'none' to disable", exc_info=e)

   
    kwargs = dict(
        x=mltk_model.x,
        y=mltk_model.y,
        batch_size=mltk_model.batch_size,
        steps_per_epoch=mltk_model.steps_per_epoch,
        validation_split=mltk_model.validation_split,
        validation_data=mltk_model.validation_data,
        validation_steps=mltk_model.validation_steps,
        validation_batch_size=mltk_model.validation_batch_size,
        validation_freq=mltk_model.validation_freq,
        shuffle=mltk_model.shuffle,
        class_weight=class_weights,
        sample_weight=mltk_model.sample_weight,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=0 if verbose == False else 1,
    )
    kwargs.update(mltk_model.train_kwargs)
    logger.debug(f'Train kwargs:\n{pprint.pformat(kwargs)}')
    
    logger.info('Starting model training ...')
    training_history = keras_model.fit(
        **kwargs
    )

    try:
        mltk_model.unload_dataset()
    except Exception as e:
        logger.debug(f'Failed to unload dataset, err: {e}', exc_info=e)


    keras_model = _save_keras_model_file(mltk_model, keras_model, logger=logger)
    results = _save_training_results(
        mltk_model, 
        keras_model, 
        training_history, 
        logger=logger,
        show=show,
    )

    if create_archive:
        _create_model_archive(mltk_model, logger)

    # Quantize the trained model
    if quantize and mltk_model.tflite_converter:
        try:
            quantize_model(
                mltk_model, 
                keras_model=results.keras_model,
                update_archive=create_archive
            )
        except Exception as e:
            prepend_exception_msg(e, 'Failed to quantize model')
            raise


    if mltk_model.on_training_complete is not None:
        try:
            mltk_model.on_training_complete(results)
        except Exception as e:
            logger.warning(f'Exception during on_training_complete() callback, err: {e}', exc_info=e)

    logger.info('Training complete')
    logger.info(f'Training logs here: {mltk_model.log_dir}')
    if create_archive:
        logger.info(f'Trained model files here: {mltk_model.archive_path}')
    logger.close()

    if show:
        plt.show(block=True)

    return results


def _get_epochs(
    mltk_model: MltkModel, 
    epochs:int, 
    callbacks:list,
    logger: logging.Logger
) -> int:
    """Update the training epochs as necessary"""
    if mltk_model.test_mode_enabled:
        logger.info('Forcing epochs=3 since test=true')
        return 3

    if epochs == -1:
        if not contains_class_type(callbacks, keras.callbacks.EarlyStopping):
            raise Exception('If mltk_model.epochs = -1 then mltk_model.early_stopping must be specified')

        logger.warning('***')
        logger.warning('*** NOTE: Setting training epochs to large value since the EarlyStopping callback is being used')
        logger.warning('***')
        epochs = 99999

    return epochs


def _get_keras_callbacks(
    mltk_model: MltkModel, 
    epochs: int, 
    logger: logging.Logger
) -> list:
    """Populate the Keras training callbacks"""
    keras_callbacks = []
    keras_callbacks.extend(mltk_model.train_callbacks)

    if mltk_model.tensorboard and not contains_class_type(keras_callbacks, keras.callbacks.TensorBoard):
        tb_log_dir = mltk_model.create_log_dir('train/tensorboard')
        kwargs = dict(log_dir=tb_log_dir)
        kwargs.update(mltk_model.tensorboard)
        logger.debug('Using default TensorBoard callback with following parameters:')
        logger.debug(f'{pprint.pformat(kwargs)}')
        cb = keras.callbacks.TensorBoard(**kwargs)
        keras_callbacks.append(cb)
        logger.info(f'Tensorboard logdir: {tb_log_dir}')
    

    if mltk_model.checkpoint and not contains_class_type(keras_callbacks, keras.callbacks.ModelCheckpoint):
        weights_dir = mltk_model.weights_dir
        weights_file_format = mltk_model.weights_file_format
        kwargs = dict(
            filepath=f'{weights_dir}/{weights_file_format}',
        )
        kwargs.update(mltk_model.checkpoint)
        logger.debug('Using default ModelCheckpoint callback with following parameters:')
        logger.debug(f'{pprint.pformat(kwargs)}')
        cb = keras.callbacks.ModelCheckpoint(**kwargs)
        keras_callbacks.append(cb)


    if mltk_model.lr_schedule and not contains_class_type(keras_callbacks, keras.callbacks.LearningRateScheduler):
        kwargs = dict()
        kwargs.update(mltk_model.lr_schedule)
        logger.debug('Using default LearningRateScheduler callback with following parameters:')
        logger.debug(f'{pprint.pformat(kwargs)}')
        cb = keras.callbacks.LearningRateScheduler(**kwargs)
        keras_callbacks.append(cb)


    if mltk_model.early_stopping and not contains_class_type(keras_callbacks, keras.callbacks.EarlyStopping):
        kwargs = dict()
        kwargs.update(mltk_model.early_stopping)
        logger.debug('Using default EarlyStopping callback with following parameters:')
        logger.debug(f'{pprint.pformat(kwargs)}')
        cb = keras.callbacks.EarlyStopping(**kwargs)
        keras_callbacks.append(cb)


    if mltk_model.reduce_lr_on_plateau and not contains_class_type(keras_callbacks, keras.callbacks.ReduceLROnPlateau):
        kwargs = dict()
        kwargs.update(mltk_model.reduce_lr_on_plateau)
        logger.debug('Using default ReduceLROnPlateau callback with following parameters:')
        logger.debug(f'{pprint.pformat(kwargs)}')
        cb = keras.callbacks.ReduceLROnPlateau(**kwargs)
        keras_callbacks.append(cb)

    if mltk_model.checkpoints_enabled:
        logger.debug('Enabling model checkpoints')
        keras_callbacks.append(keras.callbacks.ModelCheckpoint(
            filepath=mltk_model.checkpoints_dir + '/weights-{epoch:03d}.h5',
            save_weights_only=True,
            save_best_only=False,
            save_freq='epoch',
        ))

    callback_str = ', '.join([str(x.__class__.__name__) for x in keras_callbacks])
    logger.debug(f'Using Keras callbacks: {callback_str}')

    return keras_callbacks


def _try_resume_training(
    mltk_model: MltkModel, 
    keras_model: KerasModel, 
    epochs:int,
    resume_epoch: int, 
    logger: logging.Logger
) -> int:
    """Attempt to resume training at either the last available epoch or at the specified epoch

        Return initial_epoch
    """
    if resume_epoch == 0:
        return 0

    # If the --resume <epoch< option was supplied
    # then resume at the given checkpoint
    if resume_epoch+1 >= epochs:
        raise Exception(f'The resume epoch ({resume_epoch}+1) is greater than the max training epochs ({epochs})')

    checkpoint_path = mltk_model.get_checkpoint_path(resume_epoch)
    if checkpoint_path is None:
        if resume_epoch == -1:
            logger.warning('No training checkpoints found, cannot --resume. Starting from beginning')
            return 0
        
        raise Exception(f'Checkpoint not found, cannot resume training at epoch {resume_epoch}')

    fn = os.path.basename(checkpoint_path[:-len('.h5')])
    checkpoint_epoch = int(fn.split('-')[1])

    try:
        logger.info(f'Loading checkpoint weights: {checkpoint_path}')
        keras_model.load_weights(checkpoint_path)
    except Exception as e:
        prepend_exception_msg(e, f'Failed to load checkpoint weights: {checkpoint_path}')
        raise

    logger.warning(f'Resuming training at epoch {checkpoint_epoch+1} of {epochs}')

    return checkpoint_epoch


def compute_class_weights(
    mltk_model: MltkModel, 
    logger: logging.Logger
) -> dict:
    try:
        class_weights = _compute_class_weights_unsafe(mltk_model, logger=logger)
    except Exception as e:
        class_weights = None
        logger.warning("Failed to compute class weights\nSet my_model.class_weights = 'none' to disable", exc_info=e)

    return class_weights


def _compute_class_weights_unsafe(
    mltk_model: MltkModel, 
    logger: logging.Logger
) -> dict:
    """Compute the given data's class weights"""
    def _create_weights_dict():
        class_weights = mltk_model.class_weights
        if not class_weights:
            return None
        # If a dictionary where the keys directly map to the class ids was given
        # then just return the class_weights as-is
        if isinstance(class_weights, dict):
            if isinstance(list(class_weights.keys())[0], int):
                return class_weights

        # Otherwise, we need to convert the class weights from:
        # {"label1": 1.0, "label2": .5, "lable3": .4} 
        # to
        # {0: 1.0, 1: .5, 2: .4}

        try:
            class_ids = [x for x in range(len(mltk_model.classes))]
        except Exception as e:
            prepend_exception_msg(e, 
                'Class weights should be a dict with each key be an integer corresponding to a class'
            )
            raise


        if isinstance(class_weights, list):
            return dict(zip(class_ids, class_weights))
        
        if isinstance(class_weights, str):
            class_weights = class_weights.lower()
            if class_weights == 'none':
                return None 

            if class_weights not in ('balance', 'balanced'):
                raise RuntimeError(f'Invalid my_model.class_weights argument given: {class_weights}')


            if hasattr(mltk_model, 'class_counts'):
                class_counts = mltk_model.class_counts
                if 'training' in class_counts:
                    class_counts = class_counts['training']
                n_samples = sum(x for x in class_counts.values())

                if n_samples > 0:
                    n_classes = mltk_model.n_classes
                    weights = []
                    for class_name in mltk_model.classes:
                        weights.append(n_samples / (n_classes * class_counts[class_name]))

                    return dict(zip(class_ids, weights))

            y = mltk_model.y
            if y is not None:
                weights = compute_class_weight(class_weights, classes=class_ids, y=y)
                return dict(zip(class_ids, weights))

            raise RuntimeError(
                'my_model.class_weights=balanced not supported if my_model.y or mltk_model.class_counts not provided. \n'
                'Must manually set class weights in my_model.class_weights'
            )
            

        if isinstance(class_weights, dict):
            weights = {}
            for class_id, class_name in enumerate(mltk_model.classes):
                if class_name not in class_weights:
                    raise Exception(f'Class {class_name} not found in class weights')
                weights[class_id] = class_weights[class_name]

            return weights

        raise RuntimeError('Unsupported my_model.class_weight format')

    class_weights = _create_weights_dict()
    if class_weights:
        try:
            s = 'Class weights:\n'
            max_len = max([len(x) for x in mltk_model.classes])
            for class_id, class_name in enumerate(mltk_model.classes):
                s += f'{class_name.rjust(max_len)} = {class_weights[class_id]:.2f}\n'
            logger.info(s[:-1])
        except: 
            logger.info(f'Class weights: {pprint.pformat(class_weights)}')

    return class_weights
    

def _save_keras_model_file(
    mltk_model:MltkModel, 
    keras_model:KerasModel, 
    logger: logging.Logger
) -> KerasModel:
    """Save the Keras .h5 model file"""

    # If a custom model saving callback was given then invoke that now
    # So that we obtain the correct keras model
    if mltk_model.on_save_keras_model is not None:
        try:
            keras_model = mltk_model.on_save_keras_model(
                mltk_model=mltk_model, 
                keras_model=keras_model, 
                logger=logger
            )
            if keras_model is None:
                raise RuntimeError('my_model.on_save_keras_model must return a keras model instance')

        except Exception as e:
            prepend_exception_msg(e, 'Error while saving model using my_model.on_save_keras_model')
            raise
        
    # Save the keras model as a .h5 file
    try:
        h5_path = mltk_model.h5_log_dir_path
        logger.info(f'Generating {h5_path}')
        keras_model.save(h5_path, save_format='tf')
    except Exception as e:
        prepend_exception_msg(e, f'Error while saving model to {h5_path}')
        raise

    return keras_model


def _save_training_results(
    mltk_model:MltkModel, 
    keras_model:KerasModel, 
    training_history, 
    logger: logging.Logger,
    show:bool = False
) -> TrainingResults:
    """Save the training history as .json and .png"""
    results = TrainingResults(mltk_model, keras_model, training_history) 
    metric, best_val = results.get_best_metric()
    logger.info(f'\n\n*** Best training {metric} = {best_val:.3f}\n\n')

    try:
        history_json_path = f'{mltk_model.log_dir}/train/training-history.json'
        logger.debug(f'Generating {history_json_path}')
        with open(history_json_path, 'w') as f:
            json.dump(results.asdict(), f, indent=2)
    except Exception as e:
        logger.warning(f'Error while saving training results to {history_json_path}, err: {e}')


     # See https://github.com/keras-team/keras/blob/master/keras/losses.py
    supported_metrics = {}
    supported_metrics['accuracy'] = 'Accuracy'
    supported_metrics['loss'] = 'Loss'
    supported_metrics['mse'] = 'Mean Square Error'
    supported_metrics['mae'] = 'Mean Absolute Error'
    supported_metrics['mape'] = 'Mean Absolute Percentage Error'
    supported_metrics['msle '] = 'Mean Square Logarithmic Error'
    supported_metrics['bce '] = 'Binary Cross-entropy'
    supported_metrics['cce'] = 'Categorical Cross-entropy'


    found_metrics = []
    history = results.history
    
    for metric in history:
        if not metric in supported_metrics:
            continue
        if not f'val_{metric}' in history:
            continue 
        found_metrics.append(dict(
            name=metric,
            train=history[metric],
            validation=history[f'val_{metric}'],
        ))

    fig, _ = plt.subplots(figsize=(6, 6), clear=True)
    fig.suptitle(f'{mltk_model.name} Training History')
        
    # %% Plot training and validation metrics
    for i, metric in enumerate(found_metrics):
        plt.subplot(len(found_metrics), 1, i + 1)
        plt.plot(metric['train'])
        plt.plot(metric['validation'])
        plt.title(f'{supported_metrics[metric["name"]]}')
        plt.ylabel(supported_metrics[metric['name']])
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
    
    plt.subplots_adjust(hspace=.5)

    training_results_path = f'{mltk_model.log_dir}/train/training-history.png'
    logger.debug(f'Generating {training_results_path}')
    plt.savefig(training_results_path)
    if show:
        plt.show(block=False)
    else:
        fig.clear()
        plt.close(fig)

    return results


def _create_model_archive(
    mltk_model: MltkModel, 
    logger: logging.Logger
):
    logger.info(f'Creating {mltk_model.archive_path}')
    try:
        mltk_model.add_archive_dir('.', create_new=True)
        mltk_model.add_archive_file('__mltk_model_spec__')
        mltk_model.add_archive_dir('train')
        mltk_model.add_archive_dir('dataset', recursive=True)
    except Exception as e:
        logger.warning(f'Failed to generate model archive, err: {e}', exc_info=e)


def _clear_log_directory(
    mltk_model: MltkModel, 
    logger: logging.Logger, 
    recursive=False
):
    """Clear any previous training logs"""
    training_log_dir = f'{mltk_model.log_dir}/train'

    if recursive:
        logger.info(f'Cleaning all files in {mltk_model.log_dir}')
        clean_directory(mltk_model.log_dir)

    elif os.path.exists(training_log_dir):
        for fn in os.listdir(training_log_dir):
            path = f'{training_log_dir}/{fn}'
            if os.path.isfile(path):
                logger.debug(f'Removing {path}')
                try:
                    os.remove(path)
                except Exception as e:
                    logger.debug(f'Failed to remove {path}, err: {e}')

    if os.path.exists(mltk_model.log_dir):
        for fn in os.listdir(mltk_model.log_dir):
            path = f'{mltk_model.log_dir}/{fn}'
            if os.path.isfile(path):
                logger.debug(f'Removing {path}')
                try:
                    os.remove(path)
                except Exception as e:
                    logger.debug(f'Failed to remove {path}, err: {e}')

                
    if os.path.exists(mltk_model.archive_path):
        logger.debug(f'Removing {mltk_model.archive_path}')
        try:
            os.remove(mltk_model.archive_path)
        except Exception as e:
            logger.debug(f'Failed to remove {path}, err: {e}')
