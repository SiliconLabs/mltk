
from typing import Union, Callable, List
import re
import os
import logging


from mltk.utils.python import DefaultDict
from mltk.core.training_results import TrainingResults
from .base_mixin import BaseMixin
from ..model_attributes import MltkModelAttributesDecorator, CallableType, DictType
from ..model_utils import KerasModel


@MltkModelAttributesDecorator()
class TrainMixin(BaseMixin):
    """Provides training properties and methods to the base :py:class:`~MltkModel`
    
    Refer to the `Model Training <https://siliconlabs.github.io/mltk/docs/guides/model_training.html>`_ guide for more details.
    """

    @property
    def build_model_function(self) -> Callable:
        """Function that builds and returns a compiled :py:attr:`mltk.core.KerasModel` instance

        Your model definition MUST provide this setting. 

        .. highlight:: python
        .. code-block:: python

           # Create a MltkModel instance with the 'train' mixin
           class MyModel(
               MltkModel, 
               TrainMixin, 
               ImageDatasetMixin, 
               EvaluateClassifierMixin
           ):
               pass
           mltk_model = MyModel()


           # Define the model build function
           def my_model_builder(mltk_model):
               keras_model = Sequential()
               keras_model.add(Conv2D(8, kernel_size=(3,3), padding='valid', input_shape=mltk_model.input_shape))
               keras_model.add(Flatten())
               keras_model.add(Dense(mltk_model.n_classes, activation='softmax'))

               keras_model.compile(loss=mltk_model.loss, optimizer=mltk_model.optimizer, metrics=mltk_model.metrics)

               return keras_model

           # Set the MltkModel's build_model function
           mltk_model.build_model_function = my_model_builder

        .. seealso::
           * `Keras Model compile() <https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile>`_
        
        """
        return self._attributes.get_value('train.build_model_function', default=_build_model_placeholder)
    @build_model_function.setter
    def build_model_function(self, v: Callable):
        self._attributes['train.build_model_function'] = v


    @property
    def on_training_complete(self) -> Callable[[TrainingResults],None]:
        """Callback to be invoked after the model has been successfully trained

        .. highlight:: python
        .. code-block:: python
           
           def _on_training_completed(results:TrainingResults):
               ...
        
           my_model.on_training_complete = _on_training_completed

        .. note:: This is invoked after the Keras and .tflite model files are saved
        """
        return self._attributes.get_value('train.on_training_complete', default=None)
    @on_training_complete.setter
    def on_training_complete(self, v: Callable[[TrainingResults],None]):
        self._attributes['train.on_training_complete'] = v


    @property
    def on_save_keras_model(self) -> Callable[[object,KerasModel,logging.Logger],KerasModel]:
        """Callback to be invoked after the model has been trained to
        save the KerasModel. 

        This callback may be used to modified the KerasModel that gets saved,
        e.g. Remove layers of the model that were used for training.

        .. highlight:: python
        .. code-block:: python
           
           def _on_save_keras_model(mltk_model:MltkModel, keras_model:KerasModel, logger:logging.Logger) -> KerasModel:
               ...
               return keras_model
        
           my_model.on_save_keras_model = _on_save_keras_model

        .. note:: This is invoked before the model is quantized. Quantization will use the KerasModel returned by this callback.
        """
        return self._attributes.get_value('train.on_save_keras_model', default=None)
    @on_save_keras_model.setter
    def on_save_keras_model(self, v: Callable[[object,KerasModel,logging.Logger],KerasModel]):
        self._attributes['train.on_save_keras_model'] = v


    @property
    def epochs(self) -> int:
        """Number of epochs to train the model.  

        Default: ``100``

        An epoch is an iteration over the entire x and y data provided. Note that  epochs is to be understood as "final epoch". 
        The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.

        If this is set to ``-1`` then the epochs will be set to an arbitrarily large value.
        In this case, the :py:attr:`~early_stopping` calback should be used to determine
        when to stop training the model.

        .. note:: The larger this value is, the longer the model will take to train

        .. seealso::
           * `Keras Model fit() <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_

        """
        return self._attributes.get_value('train.epochs', default=100)
    @epochs.setter
    def epochs(self, v: int):
        self._attributes['train.epochs'] = v


    @property
    def batch_size(self) -> int:
        """Number of samples per gradient update  

        Default: ``32``

        Typical values are: 16, 32, 64.

        Typically, the larger this value is, the more RAM that is required during
        training.

         .. seealso::
            * `Keras Model fit() <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_
        """
        return self._attributes.get_value('train.batch_size', default=32)
    @batch_size.setter
    def batch_size(self, v: int):
        self._attributes['train.batch_size'] = v


    @property
    def optimizer(self):
        """
        String (name of optimizer) or optimizer instance  

        Default: ``adam``

        .. seealso::
        
           * `Keras Optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_
           * `Keras Model compile() <https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile>`_

        """
        return self._attributes.get_value('train.optimizer', default='adam')
    @optimizer.setter
    def optimizer(self, v):
        self._attributes['train.optimizer'] = v

    
    @property
    def metrics(self) -> List[str]:
        """List of metrics to be evaluated by the model during training and testing  

        Default: ``['accuracy']``

        .. seealso::
        
           * `Keras Metrics <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric>`_
           * `Keras Model compile() <https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile>`_
        """
        return self._attributes.get_value('train.metrics', default=['accuracy'])
    @metrics.setter
    def metrics(self, v: List[str]):
        self._attributes['train.metrics'] = v


    @property
    def loss(self):
        """String (name of objective function), objective function  
    
        Default: ``categorical_crossentropy``

        .. seealso::
        
           * `Keras Loss Functions <https://www.tensorflow.org/api_docs/python/tf/keras/losses>`_
           * `Keras Model compile() <https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile>`_

        """
        return self._attributes.get_value('train.loss', default='categorical_crossentropy')
    @loss.setter
    def loss(self, v):
        self._attributes['train.loss'] = v


    @property
    def checkpoints_enabled(self) -> bool:
        """If true, enable saving a checkpoint after each training epoch.

        Default: ``True``

        This is useful as it allows for resuming training sessions with the ``--resume``
        argument to the ``train`` command.

        .. note:: 
           This is independent of :py:attr:`~checkpoint`. 
           This saves each epoch's weights to the logdir/train/checkpoints directory regardless
           of the what's configured in :py:attr:`~checkpoint`
        
        """
        return self._attributes.get_value('train.checkpoints_enabled', default=True)
    @checkpoints_enabled.setter
    def checkpoints_enabled(self, v: bool):
        self._attributes['train.checkpoints_enabled'] = v


    @property
    def train_callbacks(self):
        """List of keras.callbacks.Callback instances.   

        Default: ``[]``
        
        List of callbacks to apply during training.  

        .. note:: 
           If a callback is found in this list, then the corresponding callback setting is ignore.  
           e.g.: If `LearningRateScheduler Callback <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler>`_
           is found in this list, then :py:attr:`~lr_schedule` is ignored.

        .. seealso::

           * `keras.callbacks.Callback <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks>`_

        """
        return self._attributes.get_value('train.callbacks', default=[])
    @train_callbacks.setter 
    def train_callbacks(self, v):
        self._attributes['train.callbacks'] = v


    @property
    def lr_schedule(self) -> dict:
        """Learning rate scheduler  

        Default: ``None``

        .. highlight:: python
        .. code-block:: python

           dict(
               schedule, # a function that takes an epoch index (integer, indexed from 0) 
                         # and current learning rate (float) as inputs and returns a new learning rate as output (float).

               verbose=0 # int. 0: quiet, 1: update messages.
           )
        
        .. note:: Set to ``None`` to disable

        At the beginning of every epoch, the this callback gets the updated learning rate value from ``schedule`` function provided, 
        with the current epoch and current learning rate, and applies the updated learning rate on the optimizer.


        .. note::

           * Set to `None` to disable this callback
           * By default, this is disabled in favor of the :py:attr:`~reduce_lr_on_plateau` callback
           * If this callback is enabled, then :py:attr:`~reduce_lr_on_plateau` is automatically disabled
        

        .. seealso::
            * `LearningRateScheduler Callback <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler>`_
        """
        return self._attributes.get_value('train.lr_schedule', default=None)
    @lr_schedule.setter
    def lr_schedule(self, v: Union[dict,Callable]):
        if isinstance(v, dict):
            self._attributes['train.lr_schedule'] = DefaultDict(v)
        else:
            self._attributes['train.lr_schedule'] = DefaultDict(
                schedule=v,
                verbose=0
            )
    

    @property
    def reduce_lr_on_plateau(self) -> dict:
        """Reduce learning rate when a metric has stopped improving

        Default:  ``None``

        Possible values:

        .. highlight:: python
        .. code-block:: python

           dict(
               monitor="val_loss",   # quantity to be monitored.
               
               factor=0.1,           # factor by which the learning rate will be reduced. new_lr = lr * factor.
               
               patience=10,          # number of epochs with no improvement after which learning rate will be reduced.
               
               mode="auto",          # one of {'auto', 'min', 'max'}. In 'min' mode, the learning rate will be reduced 
                                     # when the quantity monitored has stopped decreasing; in 'max' mode it will be reduced
                                     # when the quantity monitored has stopped increasing; in 'auto' mode, the direction is 
                                     # automatically inferred from the name of the monitored quantity.
               
               min_delta=0.0001,     # threshold for measuring the new optimum, to only focus on significant changes.
               
               cooldown=0,           # number of epochs to wait before resuming normal operation after lr has been reduced.
               
               min_lr=0,             # lower bound on the learning rate.
               
               verbose=1,            # int. 0: quiet, 1: update messages.
           )

        Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates. 
        This callback monitors a quantity and if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced.

        .. note::

           * Set to ``None`` to disable this callback
           * If :py:attr:`~lr_schedule` is enabled then this callback is automatically disabled
        

        .. seealso::
           * `ReduceLROnPlateau Callback <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau>`_
        
        """
        return self._attributes.get_value('train.reduce_lr_on_plateau', default=None)
    @reduce_lr_on_plateau.setter
    def reduce_lr_on_plateau(self, v: dict):
        self._attributes['train.reduce_lr_on_plateau'] = DefaultDict(v)


    @property
    def tensorboard(self) -> dict:
        """Enable visualizations for TensorBoard

        Default: ``None``

        Possible values:

        .. highlight:: python
        .. code-block:: python

          dict(
               histogram_freq=1,       # frequency (in epochs) at which to compute activation and weight histograms 
                                       # for the layers of the model. If set to 0, histograms won't be computed. 
                                       # Validation data (or split) must be specified for histogram visualizations.
   
               write_graph=True,       # whether to visualize the graph in TensorBoard. The log file can become quite large when write_graph is set to True.
   
               write_images=False,     # whether to write model weights to visualize as image in TensorBoard.

               update_freq="epoch",    # 'batch' or 'epoch' or integer. When using 'batch', writes the losses and metrics 
                                       # to TensorBoard after each batch. The same applies for 'epoch'. 
                                       # If using an integer, let's say 1000, the callback will write the metrics and losses 
                                       # to TensorBoard every 1000 batches. Note that writing too frequently to 
                                       # TensorBoard can slow down your training.

               profile_batch=2,        # Profile the batch(es) to sample compute characteristics. 
                                       # profile_batch must be a non-negative integer or a tuple of integers. 
                                       # A pair of positive integers signify a range of batches to profile. 
                                       # By default, it will profile the second batch. Set profile_batch=0 to disable profiling.
           ) 

        This callback logs events for `TensorBoard <https://www.tensorflow.org/tensorboard/get_started>`_, including:

        * Metrics summary plots
        * Training graph visualization
        * Activation histograms
        * Sampled profiling


        .. note::

           * Set to ``None`` to disable this callback
           * Tensorboard logs are saved to :py:attr:`mltk.core.MltkModel.log_dir`/train/tensorboard

        .. seealso::
           * `TensorBoard Callback <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard>`_

        """
        return self._attributes.get_value('train.tensorboard', default=None)
    @tensorboard.setter
    def tensorboard(self, v: dict):
        self._attributes['train.tensorboard'] = DefaultDict(v)


    @property
    def checkpoint(self) -> dict:
        """Callback to save the Keras model or model weights at some frequency

        Default:  

        .. highlight:: python
        .. code-block:: python

          dict(
               monitor="val_accuracy",   # The metric name to monitor. Typically the metrics are set by the Model.compile method. 
                                         # Note:
                                         # - Prefix the name with "val_" to monitor validation metrics.
                                         # - Use "loss" or "val_loss" to monitor the model's total loss.
                                         # - If you specify metrics as strings, like "accuracy", pass the same string (with or without the "val_" prefix).
                                         # - If you pass metrics.Metric objects, monitor should be set to metric.name
                                         # - If you're not sure about the metric names you can check the contents of the history.history dictionary returned by history = model.fit()
                                         # - Multi-output models set additional prefixes on the metric names.

               save_best_only=True,      # if save_best_only=True, it only saves when the model is considered the "best" 
                                         # and the latest best model according to the quantity monitored will not be overwritten. 
                                         # If filepath doesn't contain formatting options like {epoch} then filepath will be overwritten by each new better model.

               save_weights_only=True,   # if True, then only the model's weights will be saved (model.save_weights(filepath)), 
                                         # else the full model is saved (model.save(filepath)).

               mode="auto",              # one of {'auto', 'min', 'max'}. If save_best_only=True, the decision to overwrite 
                                         # the current save file is made based on either the maximization or the minimization of the 
                                         # monitored quantity. For val_acc, this should be max, for val_loss this should be min, etc. 
                                         # In auto mode, the direction is automatically inferred from the name of the monitored quantity.

               save_freq="epoch",        # 'epoch' or integer. When using 'epoch', the callback saves the model after each epoch. 
                                         # When using integer, the callback saves the model at end of this many batches. 
                                         # If the Model is compiled with steps_per_execution=N, then the saving criteria will be 
                                         # checked every Nth batch. Note that if the saving isn't aligned to epochs, 
                                         # the monitored metric may potentially be less reliable (it could reflect as little 
                                         # as 1 batch, since the metrics get reset every epoch). Defaults to 'epoch'.

               options=None,             # Optional tf.train.CheckpointOptions object if save_weights_only is true or optional 
                                         # tf.saved_model.SaveOptions object if save_weights_only is false.

               verbose=0,                # verbosity mode, 0 or 1.
           )

        ModelCheckpoint callback is used in conjunction with training using model.fit() 
        to save a model or weights (in a checkpoint file) at some interval, 
        so the model or weights can be loaded later to continue the training from the state saved.

        .. note::

           * Set to ``None`` to disable this callback
           * Tensorboard logs are saved to `MltkModel.log_dir`/train/weights
           * This is independent of :py:attr:`~checkpoints_enabled`. 

        .. seealso::
           * `ModelCheckpoint Callback <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint>`_
        
        """
        return self._attributes.get_value('train.checkpoint',
            default= DefaultDict(
                monitor="val_accuracy",
                save_best_only=True,
                save_weights_only=True,
                mode="auto",
                save_freq="epoch",
                options=None,
                verbose=0,
        ))
    @checkpoint.setter
    def checkpoint(self, v: dict):
        self._attributes['train.checkpoint'] = DefaultDict(v)


    @property
    def early_stopping(self) -> dict:
        """Stop training when a monitored metric has stopped improving

        Default: ``None``

        Possible values:

        .. highlight:: python
        .. code-block:: python

           dict(
               monitor="val_accuracy",     # Quantity to be monitored.

               min_delta=0,                # Minimum change in the monitored quantity to qualify as an improvement, 
                                           # i.e. an absolute change of less than min_delta, will count as no improvement.
  
               patience=25,                # Number of epochs with no improvement after which training will be stopped.

               mode="auto",                # One of {"auto", "min", "max"}. In min mode, training will stop when the quantity 
                                           # monitored has stopped decreasing; in "max" mode it will stop when the quantity monitored 
                                           # has stopped increasing; in "auto" mode, the direction is automatically inferred from 
                                           # the name of the monitored quantity.

               baseline=None,              # Baseline value for the monitored quantity. Training will stop if 
                                           # the model doesn't show improvement over the baseline.

               restore_best_weights=True,  # Whether to restore model weights from the epoch with the best value of the monitored quantity. 
                                           # If False, the model weights obtained at the last step of training are used.

               verbose=1,                  # verbosity mode.
           )

        Assuming the goal of a training is to minimize the loss. With this, the metric to be monitored would be 'loss', 
        and mode would be 'min'. A model.fit() training loop will check at end of every epoch whether the loss is 
        no longer decreasing, considering the min_delta and patience if applicable. Once it's found no longer decreasing, 
        model.stop_training is marked True and the training terminates.

        .. note::
           * Set to ``None`` to disable this callback
           * Set :py:attr:`~epochs` to ``-1`` to always train until early stopping is triggered

        .. seealso::
           * `EarlyStopping Callback <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping>`_

        """
        return self._attributes.get_value('train.early_stopping', default=None)
    @early_stopping.setter
    def early_stopping(self, v: dict):
        self._attributes['train.early_stopping'] = DefaultDict(v)


    @property
    def tflite_converter(self) -> dict:
        """Converts a TensorFlow model into TensorFlow Lite model  

        Default:

        .. highlight:: python
        .. code-block:: python

           dict(
               optimizations = [tf.lite.Optimize.DEFAULT],             # Experimental flag, subject to change. 
                                                                       # A list of optimizations to apply when converting the model. E.g. [Optimize.DEFAULT]
   
               supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8],  # Experimental flag, subject to change. Set of OpsSet options supported by the device. 
                                                                       # Add to the 'target_spec' option
                                                                       # https://www.tensorflow.org/api_docs/python/tf/lite/TargetSpec

               inference_input_type = tf.float32,                      # Data type of the input layer. Note that integer types (tf.int8 and tf.uint8) are 
                                                                       # currently only supported for post training integer quantization and quantization aware training. 
                                                                       # (default tf.float32, must be in {tf.float32, tf.int8, tf.uint8})

               inference_output_type = tf.float32,                     # Data type of the output layer. Note that integer types (tf.int8 and tf.uint8) are currently only 
                                                                       # supported for post training integer quantization and quantization aware training.
                                                                       # (default tf.float32, must be in {tf.float32, tf.int8, tf.uint8})

               representative_dataset = 'generate',                    # A representative dataset that can be used to generate input and output samples 
                                                                       # for the model. The converter can use the dataset to evaluate different optimizations. 
                                                                       # Note that this is an optional attribute but it is necessary if INT8 is the only 
                                                                       # support builtin ops in target ops.
                                                                       # If the keyword 'generate' is used, then use update to 1000 samples from the model's
                                                                       # validation dataset as the representative dataset

               allow_custom_ops = False,                               # Boolean indicating whether to allow custom operations. When False, any unknown operation is an error. 
                                                                       # When True, custom ops are created for any op that is unknown. The developer needs to provide these to the 
                                                                       # TensorFlow Lite runtime with a custom resolver. (default False)

               experimental_new_converter = True,                      # Experimental flag, subject to change. Enables MLIR-based conversion instead of TOCO conversion. (default True)

               experimental_new_quantizer = True,                      # Experimental flag, subject to change. Enables MLIR-based quantization conversion instead of Flatbuffer-based conversion. (default True)

               experimental_enable_resource_variables = False,         # Experimental flag, subject to change. Enables resource variables to be converted by this converter. 
                                                                       # This is only allowed if from_saved_model interface is used. (default False)

               generate_unquantized = True                             # Also generate a float32/unquantized .tflite model in addition to the quantized .tflite model
           )

        This is used after the model finishes training.
        The trained Keras .h5 model file is converted to a .tflite file using the TFLiteConverter
        using the settings specified by this field.

        If ``generate_unquantized=True`` then a quantized .tflite AND an unquantized .tflite model files with be generated.
        If you ONLY want to generate an unquantized model, then  ``supported_ops = TFLITE_BUILTINS``

        .. note:: 
           See :py:attr:`~on_training_complete` to invoke a custom callback which may be
           used to perform custom quantization

        .. seealso::
           * `Model Quantization <https://siliconlabs.github.io/mltk/docs/guides/model_quantization.html>`_ guide
           * `tf.lite.TFLiteConverter <https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter>`_
           * `Post-training integer quantization <https://www.tensorflow.org/lite/performance/post_training_integer_quant>`_
        """
        return self._attributes.get_value('train.tflite_converter', 
            default=DefaultDict(
                optimizations = ['DEFAULT'],
                supported_ops = ['TFLITE_BUILTINS_INT8'],
                inference_input_type = 'float32',
                inference_output_type = 'float32',
                representative_dataset = 'generate',
                allow_custom_ops = False, 
                experimental_new_converter = True, 
                experimental_new_quantizer = True, 
                experimental_enable_resource_variables = False,
                generate_unquantized = True
        ))
    @tflite_converter.setter
    def tflite_converter(self, v: dict):
        self._attributes['train.tflite_converter'] = DefaultDict(v)


    @property 
    def checkpoints_dir(self) -> str:
        """Return path to directory containing training checkpoints"""
        return self.create_log_dir('train/checkpoints')


    def get_checkpoint_path(self, epoch=None) -> str:
        """Return the file path to the checkpoint weights for the given epoch

        If no epoch is provided then return the best checkpoint weights file is return.
        Return None if no checkpoint is found.

        .. note:: Checkpoints are only generated if :py:attr:`~checkpoints_enabled` is True.
        """

        # If not epoch is specified,
        # then find the last checkpoint generated
        if epoch is None:
            epoch = -1 

        checkpoints_dir = self.checkpoints_dir
        
        name_re = re.compile(r'^weights-(\d+)\.h5$')
        found_checkpoint_name = None 
        found_checkpoint_epoch = 0 

        for fn in os.listdir(checkpoints_dir):
            match = name_re.match(fn)
            if match is None:
                continue 
            checkpoint_epoch = int(match.group(1))

            if epoch == -1:
                if checkpoint_epoch > found_checkpoint_epoch:
                    found_checkpoint_name = fn 
                    found_checkpoint_epoch = checkpoint_epoch
                    
            elif checkpoint_epoch == epoch:
                found_checkpoint_name = fn 
                found_checkpoint_epoch = checkpoint_epoch
                break

        if found_checkpoint_name is None:
            return None

        return f'{checkpoints_dir}/{found_checkpoint_name}'


    @property 
    def weights_dir(self) -> str:
        """Return path to directory contianing training weights"""
        return self.create_log_dir('train/weights')

    @property
    def weights_file_format(self) -> str:
        """Return the file format used to generate model weights files during training"""

        if self.checkpoint is None:
            raise Exception('Must specify "checkpoint" attribute')
        monitor = self.checkpoint['monitor']
        return 'weights-{epoch:03d}-{%s:.4f}.h5' % monitor


    def get_weights_path(self, filename: str=None) -> str:
        """Return the path to a Keras .h5 weights file"""

        if isinstance(filename, str):
            if os.path.exists(filename):
                return filename 

        weights_dir = self.weights_dir

        if isinstance(filename, str):
            if os.path.exists(f'{weights_dir}/{filename}'):
                return f'{weights_dir}/{filename}'

        if not (filename is None or filename == 'best'):
            raise Exception(f'Weights file not found: {filename}')

        best_weights_path = None 
        best_weights = 0.0
        for fn in os.listdir(weights_dir):
            if not fn.endswith('.h5'):
                continue
            toks = fn[:-3].split('-')
            if len(toks) != 3:
                continue 
            try:
                weights = float(toks[2])
                if weights > best_weights:
                    best_weights = weights
                    best_weights_path = f'{weights_dir}/{fn}'
            except:
                pass 

        if best_weights_path is None:
            raise Exception('Best training weights not found')

        return best_weights_path

    @property
    def train_kwargs(self) -> dict:
        """Additional arguments to pass the the `model fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_ API.
        These keyword arguments will override the other model properties passed to fit().
        """
        return self._attributes.get_value('train.kwargs', default={})
    @train_kwargs.setter
    def train_kwargs(self, v:dict):
        self._attributes['train.kwargs'] = v


    def _register_attributes(self):
        self._attributes.register('train.build_model_function', dtype=CallableType)
        self._attributes.register('train.epochs', dtype=int)
        self._attributes.register('train.batch_size', dtype=int)
        self._attributes.register('train.optimizer')
        self._attributes.register('train.metrics', dtype=(list,tuple))
        self._attributes.register('train.loss', dtype=(str,CallableType))
        self._attributes.register('train.checkpoints_enabled', dtype=bool)
        self._attributes.register('train.callbacks', dtype=(list,tuple))
        self._attributes.register('train.lr_schedule', dtype=(DictType, CallableType))
        self._attributes.register('train.reduce_lr_on_plateau', dtype=DictType)
        self._attributes.register('train.tensorboard', dtype=DictType)
        self._attributes.register('train.checkpoint', dtype=DictType)
        self._attributes.register('train.early_stopping', dtype=DictType)
        self._attributes.register('train.tflite_converter', dtype=DictType)
        self._attributes.register('train.on_training_complete', dtype=CallableType)
        self._attributes.register('train.on_save_keras_model', dtype=CallableType)
        self._attributes.register('train.kwargs', dtype=dict)


def _build_model_placeholder(mltk_model):
    raise NotImplementedError(
        'You model must define the property "mltk_model.build_model_function" which '
        'points to function that returns a compiled Keras model'
    )