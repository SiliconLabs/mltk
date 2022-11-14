
from typing import List, Tuple
import logging
import os
import json

import numpy as np
import tensorflow as tf
import sklearn

import matplotlib.pyplot as plt
from mltk.utils import gpu
from mltk.utils.python import prepend_exception_msg
from .model import (
    MltkModel,
    KerasModel,
    TrainMixin,
    DatasetMixin,
    EvaluateAutoEncoderMixin,
    load_tflite_or_keras_model
)
from .utils import get_mltk_logger
from .summarize_model import summarize_model
from .evaluation_results import EvaluationResults


class AutoEncoderEvaluationResults(EvaluationResults):
    """Auto-encoder evaluation results
    
    .. seealso::
    
       - :py:func:`~evaluate_autoencoder`
       - :py:func:`mltk.core.evaluate_model`
    
    """
    def __init__(self, *args, **kwargs):
        EvaluationResults.__init__(self, *args,  model_type='auto_encoder', **kwargs)

    @property
    def classes(self) -> List[str]:
        """List of class labels used by evaluated model"""
        return self['classes']

    @property
    def overall_accuracy(self) -> float:
        """The overall, model accuracy"""
        return self['overall_accuracy']

    @property
    def overall_precision(self) -> List[float]:
        """The overall, model precision as various thresholds"""
        return self['overall_precision']

    @property
    def overall_recall(self) -> List[float]:
        """The overall, model recall at various thresholds"""
        return self['overall_recall']

    @property
    def overall_pr_accuracy(self) -> float:
        """The overall, precision vs recall"""
        return self['overall_pr_accuracy']


    @property
    def overall_tpr(self) -> List[float]:
        """The overall, true positive rate at various thresholds"""
        return self['overall_tpr']

    @property
    def overall_fpr(self) -> List[float]:
        """The overall, false positive rate at various thresholds"""
        return self['overall_fpr']

    @property
    def overall_roc_auc(self) -> List[float]:
        """The overall, area under curve of the receiver operating characteristic"""
        return self['overall_roc_auc']

    @property
    def overall_thresholds(self) -> List[float]:
        """List of thresholds used to calcuate overall stats"""
        return self['overall_thresholds']

    @property
    def class_stats(self) -> dict:
        """Dictionary of per class statistics"""
        return self['class_stats']


    def calculate(
        self, 
        y:np.ndarray, 
        y_pred:np.ndarray,
        all_scores: np.ndarray,
        thresholds: List[float] = None
    ):
        """Calculate the evaluation results
        
        Given the list of expected values and corresponding predicted values with scores, 
        calculate the evaluation metrics.
        
        Args:
            y: 1D array of expected class ids
            y_pred: 1D array of scoring results, e.g. y_pred[i] = scoring_function(x[i], y[i])
            all_scores: 2D [n_samples, n_classes] of scores comparing the input vs auto-encoder generated out for each class type (normal, and all abnormal cases)
            thresholds: Optional, list of thresholds to use for calculating the TPR, FPR and AUC
        """

        if thresholds is None:
            thresholds = [x for x in np.amin(y_pred) + np.arange(0.0, 1.01, .01)*(np.amax(y_pred)-np.amin(y_pred))]

        self['all_scores'] = all_scores
        self['thresholds'] = thresholds
        self['overall_accuracy'] = calculate_overall_accuracy(y_pred, y)
        self['overall_precision'], self['overall_recall'], self['overall_pr_accuracy'] = calculate_overall_pr_accuracy(thresholds, y_pred, y)
        self['overall_tpr'], self['overall_fpr'], self['overall_roc_auc'] = calculate_overall_roc_auc(thresholds, y_pred, y)
        self['class_stats'] = calculate_class_stats(all_scores, self['classes'])


    def generate_summary(self) -> str:
        """Generate and return a summary of the results as a string"""
        s = super().generate_summary(include_all=False)
        return s + '\n' + summarize_results(self)

    def generate_plots(
        self, 
        show=True, 
        output_dir:str=None, 
        logger: logging.Logger=None
    ):
        """Generate plots of the evaluation results
        
        Args:
            show: Display the generated plots
            output_dir: Generate the plots at the specified directory. If omitted, generated in the model's logging directory
            logger: Optional logger
        """
        plot_results(
            self, 
            logger=logger, 
            output_dir=output_dir, 
            show=show
        )



def evaluate_autoencoder(
    mltk_model:MltkModel,
    tflite:bool=False,
    weights:str=None,
    max_samples_per_class:int=-1,
    classes:List[str]=None,
    dump: bool=False,
    verbose: bool=None,
    show: bool=False,
    callbacks:list=None,
    update_archive:bool=True
) -> AutoEncoderEvaluationResults:
    """Evaluate a trained auto-encoder model
    
    Args:
        mltk_model: MltkModel instance
        tflite: If true then evalute the .tflite (i.e. quantized) model, otherwise evaluate the keras model
        weights: Optional weights to load before evaluating (only valid for a keras model)
        max_samples_per_class: Maximum number of samples per class to evaluate. This is useful for large datasets
        classes: Specific classes to evaluate, if omitted, use the one defined in the given MltkModel, i.e. model specification
        dump: If true, dump the model output of each sample with a side-by-side comparsion to the input sample
        verbose: Enable verbose log messages
        show: Show the evaluation results diagrams
        callbacks: Optional callbacks to invoke while evaluating
        update_archive: Update the model archive with the eval results

    Returns:
        Dictionary containing evaluation results
    """

    if not isinstance(mltk_model, TrainMixin):
        raise Exception('MltkModel must inherit TrainMixin')
    if not isinstance(mltk_model, EvaluateAutoEncoderMixin):
        raise Exception('MltkModel must inherit EvaluateAutoEncoderMixin')
    if not isinstance(mltk_model, DatasetMixin):
        raise Exception('MltkModel must inherit a DatasetMixin')

    subdir = 'eval/tflite' if tflite else 'eval/h5'
    eval_dir = mltk_model.create_log_dir(subdir, delete_existing=True)
    dump_dir = mltk_model.create_log_dir(f'{subdir}/dumps')
    logger = mltk_model.create_logger('eval', parent=get_mltk_logger())

    gpu.initialize(logger=logger)
    if update_archive:
        update_archive = mltk_model.check_archive_file_is_writable()

    scoring_function = mltk_model.get_scoring_function()
    classes = classes or mltk_model.eval_classes

    # Build the MLTK model's corresponding as a Keras model or .tflite
    try:
        built_model = load_tflite_or_keras_model(
            mltk_model, 
            model_type='tflite' if tflite else 'h5',
            weights=weights
        )
    except Exception as e:
        prepend_exception_msg(e, 'Failed to build model')
        raise

    try:
        summary = summarize_model(
            mltk_model,
            built_model=built_model
        )
        logger.info(summary)
    except Exception as e:
        logger.debug(f'Failed to generate model summary, err: {e}', exc_info=e)
        logger.warning(f'Failed to generate model summary, err: {e}')

    logger.info('Evaluating auto-encoder model ...')
    
    all_scores = []
    for class_label in classes:
        logger.info(f'Loading dataset for class: {class_label}')

        try:
            mltk_model.load_dataset(
                subset='evaluation', 
                max_samples_per_class=max_samples_per_class,
                classes=[class_label],
                logger=logger,
                test=mltk_model.test_mode_enabled
            )
        except Exception as e:
            prepend_exception_msg(e, 'Failed to load model evaluation dataset' )
            raise

        eval_data = _retrieve_data(mltk_model.x)

        logger.info(f'Generating model predictions for {class_label} class ...')
        if isinstance(built_model, KerasModel):
            y_pred = built_model.predict(
                x = eval_data, 
                callbacks=callbacks,
                verbose=1 if verbose else 0,
            )
        else:
            y_pred = built_model.predict(x = eval_data, y_dtype=np.float32)

        # loop over all original images and their corresponding reconstructions
        class_scores = np.empty((len(eval_data),), dtype=np.float32)
        dump_count = 0
        for i, (orig, decoded) in enumerate(zip(eval_data, y_pred)):
            try:
                class_scores[i] = scoring_function(orig, decoded)
            except Exception as e:
                prepend_exception_msg(e, 'Error executing scoring function')
                raise
                
            if dump and dump_count < 200: # Don't dump more than 200 samples
                dump_count += 1
                dump_path = f'{dump_dir}/{class_label}/{i}.png'
                _save_decoded_image(dump_path, orig, decoded, class_scores[i])
        
        all_scores.append(class_scores)
    
    mltk_model.unload_dataset()

    if dump:
        logger.info(f'Decoded comparisons available at {dump_dir}')


    normal_pred = all_scores[0]
    for i in range(1, len(all_scores)):
        abnormal_scores = all_scores[i]
        y_pred = np.append(normal_pred, abnormal_scores)
        y_true = np.append(np.zeros_like(normal_pred), np.ones_like(abnormal_scores))
            
    results = AutoEncoderEvaluationResults(
        name= mltk_model.name,
        classes=classes,
    )
    results.calculate(
        y = y_true,
        y_pred = y_pred,
        all_scores = all_scores
    )

    summarized_results = results.generate_summary()
    
    eval_results_path = f'{eval_dir}/eval-results.json'
    with open(eval_results_path, 'w') as f:
        json.dump(results, f, default=_encode_ndarray)
    logger.debug(f'Generated {eval_results_path}')

    summary_path = f'{eval_dir}/summary.txt'
    with open(summary_path, 'w') as f:
        f.write(summarized_results)
    logger.debug(f'Generated {summary_path}')

    results.generate_plots(
        logger=logger, 
        output_dir=eval_dir, 
        show=show
    )
    
    if update_archive:
        try:
            logger.info(f'Updating {mltk_model.archive_path}')
            mltk_model.add_archive_dir(subdir)
        except Exception as e:
            logger.warning(f'Failed to add eval results to model archive, err: {e}', exc_info=e)

    logger.close() # close the eval logger

    if show:
        plt.show(block=True)

    return results





def summarize_results(results: AutoEncoderEvaluationResults) -> str:
    """Generate a summary of the evaluation results"""

    s = '' 
    s += 'Overall accuracy: {:.3f}%\n'.format(results['overall_accuracy'] * 100)
    s += 'Precision/recall accuracy: {:.3f}%\n'.format(results['overall_pr_accuracy'] * 100)
    s += 'Overall ROC AUC: {:.3f}%\n'.format(results['overall_roc_auc'] * 100)

    if len(results['class_stats']) > 1:
        s += 'Individual class ROC AUC:\n'
        for class_label, stats in results['class_stats'].items():
            s += '  {}: {:.3f}%\n'.format(class_label, stats['auc'] * 100)

    return s.strip()


    
def plot_results(results:AutoEncoderEvaluationResults, show=False, output_dir:str=None, logger: logging.Logger=None):
    """Use Matlibplot to generate plots of the evaluation results"""

    plot_overall_roc(results, output_dir=output_dir, show=show, logger=logger)
    plot_overall_precision_vs_recall(results, output_dir=output_dir, show=show, logger=logger)
    plot_histogram(results, output_dir=output_dir, show=show, logger=logger)
    plot_class_roc(results, output_dir=output_dir, show=show, logger=logger)

    if show:
        plt.show(block=True)




def calculate_overall_accuracy(y_pred, y_true) -> float:
    """ Classifier overall accuracy calculation
    y_pred contains the outputs of the network for the validation data
    y_true are the correct answers (0.0 for normal, 1.0 for anomaly)
    """
    thresholds = np.amin(y_pred) + np.arange(0.0, 1.0, .01)*(np.amax(y_pred)-np.amin(y_pred))
    accuracy = 0.0

    for threshold in thresholds:
        y_pred_binary = (y_pred > threshold).astype(int)
        correct = np.sum(y_pred_binary == y_true)
        accuracy_tmp = correct / len(y_pred_binary)
        if accuracy_tmp > accuracy:
            accuracy = accuracy_tmp

    return accuracy      


def calculate_overall_pr_accuracy(thresholds, y_pred, y_true) -> Tuple[List[float], List[float], float]:
    """Classifier overall accuracy calculation
    y_pred contains the outputs of the network for the validation data
    y_true are the correct answers (0.0 for normal, 1.0 for anomaly)
    this is the function that should be used for accuracy calculations
    """
    # initialize all arrays
    accuracy = 0
    n_normal = np.sum(y_true == 0)
    precision = [0.0 for _ in range(len(thresholds))]
    recall = [0.0 for _ in range(len(thresholds))]

    # Loop on all the threshold values
    for threshold_item in range(len(thresholds)):
        threshold = thresholds[threshold_item]
        # Binarize the result
        y_pred_binary = (y_pred > threshold).astype(int)
        # Build matrix of TP, TN, FP and FN
        #true_negative = np.sum((y_pred_binary[0:n_normal] == 0))
        false_positive = np.sum((y_pred_binary[0:n_normal] == 1))
        true_positive = np.sum((y_pred_binary[n_normal:] == 1))
        false_negative = np.sum((y_pred_binary[n_normal:] == 0))
        # Calculate and store precision and recall
        precision[threshold_item] = true_positive / max(true_positive+false_positive, 1e-9)
        recall[threshold_item] = true_positive / max(true_positive+false_negative, 1e-9)
        # See if the accuracy has improved
        accuracy_tmp = (precision[threshold_item]+recall[threshold_item]) / 2
        if accuracy_tmp > accuracy:
            accuracy = accuracy_tmp

    return precision, recall, accuracy 


def calculate_overall_roc_auc(thresholds, y_pred, y_true) -> Tuple[List[float], List[float], float]:
    """Autoencoder ROC AUC calculation
    y_pred contains the outputs of the network for the validation data
    y_true are the correct answers (0.0 for normal, 1.0 for anomaly)
    this is the function that should be used for accuracy calculations
    """
    # initialize all arrays
    roc_auc = 0

    n_normal = np.sum(y_true == 0)
    tpr = [0.0 for _ in range(len(thresholds))]
    fpr = [0.0 for _ in range(len(thresholds))]

    # Loop on all the threshold values
    for threshold_item in range(1,len(thresholds)):
        threshold = thresholds[threshold_item]
        # Binarize the result
        y_pred_binary = (y_pred > threshold).astype(int)
        # Build TP and FP
        tpr[threshold_item] = np.sum((y_pred_binary[n_normal:] == 1))/float(len(y_true)-n_normal)
        fpr[threshold_item] = np.sum((y_pred_binary[0:n_normal] == 1))/float(n_normal)

        # Force boundary condition
        fpr[0] = 1
        tpr[0] = 1

    # Integrate
    for threshold_item in range(len(thresholds)-1):
        roc_auc += .5*(tpr[threshold_item]+tpr[threshold_item+1])*(fpr[threshold_item]-fpr[threshold_item+1])

    return tpr, fpr, roc_auc


def calculate_class_stats(all_scores, classes) -> dict:
    """Calculate stats for individual stats of each class"""
    from sklearn.metrics import (precision_recall_curve, roc_curve, auc) # pylint: disable=import-outside-toplevel

    stats = {}
    normal_pred = all_scores[0]
    total_scores = len(normal_pred)
    for i in range(1, len(all_scores)):
        abnormal_scores = all_scores[i]
        total_scores += len(abnormal_scores)
        y_pred = np.append(normal_pred, abnormal_scores)
        y_true = np.append(np.zeros_like(normal_pred), np.ones_like(abnormal_scores))
        
        fpr, tpr, thr = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        
        stats[classes[i]] = \
        {
            'fpr': fpr,
            'tpr': tpr,
            'thr': thr,
            'auc': roc_auc,
            'precision': precision,
            'recall': recall
        }

    # If more than 2 classes were provided then generate a stat for:
    # normal + <all other classes>
    if len(classes) > 2:
        y_pred = np.empty((total_scores,), dtype=np.float32)
        y_true = np.empty((total_scores,), dtype=np.int32)
        offset = 0
        for i, class_scores in enumerate(all_scores):
            n_samples = len(class_scores)
            y_pred[offset : offset + n_samples] = class_scores
            if i == 0:
                y_true[offset : offset + n_samples] = np.zeros_like(class_scores)
            else:
                y_true[offset : offset + n_samples] = np.ones_like(class_scores)
                
            offset += n_samples
        
        fpr, tpr, thr = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        stats['all'] = \
        {
            'fpr': fpr,
            'tpr': tpr,
            'thr': thr,
            'auc': roc_auc,
            'precision': precision,
            'recall': recall
        }

    return stats


def plot_overall_roc(results, output_dir:str, show:bool, logger: logging.Logger):
    """Generate a plot of the AUC ROC evaluation results"""
    name = results['name']

    fpr = results['overall_fpr']
    tpr = results['overall_tpr']
    roc_auc = results['overall_roc_auc']

    title = f'Overall ROC: {name}'
    fig = plt.figure(title)
    plt.plot(fpr, tpr, label=f"auc: {roc_auc:0.3f}")

    plt.xlim([0.0, 0.1])
    plt.ylim([0.00, 1.01])
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.grid(which='major')
    
    if output_dir:
        output_path = output_dir + f'/{name}-overall_roc.png'
        plt.savefig(output_path)
        logger.debug(f'Generated {output_path}')

    if show:
        plt.show(block=False)
    else:
        fig.clear()
        plt.close(fig)



def plot_overall_precision_vs_recall(results: dict, output_dir:str, show, logger: logging.Logger):
    """Generate a plot of the precision vs recall"""
    
    name = results['name']
    precision = results['overall_precision']
    recall = results['overall_recall']

    title = f'Precision vs Recall: {name}'
    fig = plt.figure(title)

    plt.plot(recall, precision)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.grid()

    if output_dir:
        output_path = output_dir + f'/{name}-overall_precision_vs_recall.png'
        plt.savefig(output_path)
        logger.debug(f'Generated {output_dir}')
       
    if show:
        plt.show(block=False)
    else:
        fig.clear()
        plt.close(fig)


def plot_histogram(results: dict, output_dir:str, show, logger: logging.Logger):
    """Generate a historgram image diagram from the evaluation scores"""

    name = results['name']
    all_scores = results['all_scores']
    classes = results['classes']

    min_pred = 1e10
    max_pred = -1e10
    for scores in all_scores:
        if min(scores) < min_pred:
            min_pred = min(scores)
        if max(scores) > max_pred:
            max_pred = max(scores)

    fig, ax = plt.subplots(2,1,figsize=(10,5))
    plt.subplots_adjust(hspace=.4)
     
    ax[0].set_title('Loss')
    for i, class_scores in enumerate(all_scores):
        ax[0].plot(class_scores, label=classes[i])

    ax[0].set_xlabel('Sample index')
    ax[0].set_ylabel('Predicted value')
    ax[0].legend()
    ax[0].grid()

    ax[1].set_title('Histogram')
    kwargs = dict(histtype='stepfilled', alpha=0.5, density=True, range=[min_pred, max_pred], bins=100)
    for i, class_scores in enumerate(all_scores):
        ax[1].hist(class_scores,**kwargs, label=classes[i])

    ax[1].set_xlabel('Predicted value')
    ax[1].set_ylabel('Probability')
    ax[1].legend()

    if output_dir:
        output_path = output_dir + f'/{name}-histogram.png'
        plt.savefig(output_path)
        logger.debug(f'Generated {output_dir}')

    if show:
        plt.show(block=False)
    else:
        fig.clear()
        plt.close(fig)



def plot_class_roc(results:dict, output_dir:str, show, logger: logging.Logger):
    """Generate a plot of the AUC ROC evaluation results"""
    name = results['name']
    classes = results['classes']
    class_stats = results['class_stats']

    fig, ax = plt.subplots(2,1,figsize=(10,10))
     
    ax[0].set_title(f'ROC: {name}')
    for class_label, stat in class_stats.items():
        auc = stat['auc']
        if len(classes) > 2: 
            label=f'AUC {class_label}: {auc:0.4f}'
        else: 
            label=f'AUC: {auc:0.4f}'
        ax[0].plot(stat['fpr'], stat['tpr'], label=label)
    
    ax[0].set_xlim([0.0, 1.0])
    ax[0].set_ylim([0.0, 1.01])
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].legend(loc="lower right")
    ax[0].grid()
    
    ax[1].set_title('Precision vs Recall')
    for class_label, stat in class_stats.items():
        ax[1].plot(stat['recall'], stat['precision'], label=class_label)
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_ylim([0.0, 1.01])
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    if len(classes) > 2:
        ax[1].legend()
    ax[1].grid()
    

    if output_dir:
        output_path = output_dir + f'/{name}-class_roc.png'
        plt.savefig(output_path)
        logger.debug(f'Generated {output_path}')
    if show:
        plt.show(block=False)
    else:
        fig.clear()
        plt.close(fig)






def _retrieve_data(x):
    if isinstance(x, np.ndarray):
        return x 
    if isinstance(x, tf.Tensor):
        return x.numpy()

    data = []
    if hasattr(x, 'max_samples') and x.max_samples > 0:
        max_samples = x.max_samples
    elif hasattr(x, 'samples') and x.samples > 0:
        max_samples = x.samples
    else:
        max_samples = 10000

    for batch_x, _ in x:
        if len(data) >= max_samples: 
            break
        for sample in batch_x:
            data.append(sample) 
            if len(data) >= max_samples: 
                break

    try:
        x.reset()
    except:
        pass

    return np.array(data)


def _save_decoded_image(out_path, orig, decoded, score):
    # pylint: disable=no-member
    try:
        from cv2 import cv2 
    except:
        try:
            import cv2
        except:
            raise RuntimeError('Failed import cv2 Python package, try running: pip install opencv-python OR pip install silabs-mltk[full]')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    shape = orig.shape
    if len(shape) == 1:
        plt.plot(orig, 'b')
        plt.plot(decoded, 'r')
        plt.fill_between(np.arange(shape[0]), decoded, orig, color='lightcoral')
        plt.legend(labels=["Input", "Reconstruction", "Error"])
        plt.suptitle('Score: {:1.7f}'.format(abs(score)))
        plt.savefig(out_path)
        plt.clf()
        plt.close()

    elif len(shape) == 2 or len(shape) == 3:
        img1 = sklearn.preprocessing.minmax_scale(orig.ravel(), feature_range=(0,255)).reshape(shape)
        img2 = sklearn.preprocessing.minmax_scale(decoded.ravel(), feature_range=(0,255)).reshape(shape)
        
        
        # stack the original and reconstructed image side-by-side
        output = np.hstack([img1, img2])
        outputs = cv2.applyColorMap(output.astype(np.uint8), cv2.COLORMAP_HOT)
        scale_factor = 200 / outputs.shape[1]
        width = int(outputs.shape[1] * scale_factor)
        height = int(outputs.shape[0] * scale_factor)
        outputs = cv2.resize(outputs, (width, height), interpolation = cv2.INTER_AREA)
        outputs = cv2.putText(outputs, 
                            text='Score: {:1.7f}'.format(abs(score)), 
                            org=(1, 12), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale=.5,
                            color=(0,255,0))
        cv2.imwrite(out_path, outputs)
    else:
        raise RuntimeError('Data shape not supported')


def _encode_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    raise TypeError(repr(object) + " is not JSON serialized")