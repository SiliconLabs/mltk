from typing import List, Tuple,Union
import logging
import json

import tqdm
import tensorflow as tf
from sklearn.metrics import (precision_recall_curve, confusion_matrix)
from sklearn.preprocessing import label_binarize
import numpy as np 
import matplotlib.pyplot as plt
from mltk.utils import gpu
from mltk.utils.python import prepend_exception_msg
from .model import (
    MltkModel,
    KerasModel,
    TrainMixin, 
    DatasetMixin,
    EvaluateClassifierMixin,
    load_tflite_or_keras_model
)
from .utils import get_mltk_logger
from .summarize_model import summarize_model
from .evaluation_results import EvaluationResults



class ClassifierEvaluationResults(EvaluationResults):
    """Classifier evaluation results
    
    .. seealso::

       - :py:func:`~evaluate_classifier`
       - :py:func:`mltk.core.evaluate_model`
    
    """

    def __init__(self, *args, **kwargs):
        EvaluationResults.__init__(self, *args,  model_type='classification', **kwargs)

    @property
    def classes(self) -> List[str]:
        """List of class labels used by evaluated model"""
        return self['classes']

    @property
    def overall_accuracy(self) -> float:
        """The overall, model accuracy"""
        return self['overall_accuracy']

    @property
    def class_accuracies(self) -> List[float]:
        """List of each classes' accuracy"""
        return self['class_accuracies']

    @property
    def false_positive_rate(self) -> float:
        """The false positive rate"""
        return self['fpr']

    @property
    def fpr(self) -> float:
        """The false positive rate"""
        return self['fpr']

    @property
    def tpr(self) -> float:
        """The true positive rate"""
        return self['tpr']

    @property
    def roc_auc(self) -> List[float]:
        """The area under the curve of the Receiver operating characteristic for each class"""
        return self['roc_auc']

    @property
    def roc_thresholds(self) -> List[float]:
        """The list of thresholds used to calculate the Receiver operating characteristic"""
        return self['roc_thresholds']

    @property
    def roc_auc_avg(self) -> List[float]:
        """The average of each classes' area under the curve of the Receiver operating characteristic"""
        return self['roc_auc_avg']

    @property
    def precision(self) -> List[List[float]]:
        """List of each classes' precision at various thresholds"""
        return self['precision']

    @property
    def recall(self) -> List[List[float]]:
        """List of each classes' recall at various thresholds"""
        return self['recall'] 

    @property
    def confusion_matrix(self) -> List[List[float]]:
        """Calculated confusion matrix"""
        return self['confusion_matrix'] 


    def calculate(self, y: Union[np.ndarray,list], y_pred: Union[np.ndarray,list]):
        """Calculate the evaluation results
        
        Given the expected y values and corresponding predictions,
        calculate the various evaluation results

        Args:
            y: 1D array with shape [n_samples] where each entry is the expected class label (aka id) for the corresponding sample
                e.g. 0 = cat, 1 = dog, 2 = goat, 3 = other
            y_pred: 2D array as shape [n_samples, n_classes] for categorical or 1D array as [n_samples] for binary, 
                where each entry contains the model output for the given sample.
                For binary, the values must be between 0 and 1 where < 0.5 maps to class 0 and >= 0.5 maps to class 1
        """
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.asarray(y_pred)

        if len(y_pred.shape) == 2:
            if y_pred.shape[1] == 1:
                y_pred = np.squeeze(y_pred, -1)

        if len(y_pred.shape) == 1:
           n_classes = 2
           n_samples = len(y_pred)
           y_pred_orig = y_pred
           y_pred = np.zeros((n_samples, n_classes), dtype=np.float32)
           for i, pred in enumerate(y_pred_orig):
               class_id = 0 if pred < 0.5 else 1
               y_pred[i][class_id] = pred
        else:
            n_classes = y_pred.shape[1]

        if 'classes' not in self or not self['classes']:
            self['classes'] = [str(x) for x in range(n_classes)]

        if len(y.shape) == 2:
            if y.shape[1] == 1:
                y = np.squeeze(y, -1)

        assert len(y) == len(y_pred), 'y and y_pred must have same number of samples'

        self['overall_accuracy'] = calculate_overall_accuracy(y_pred, y)
        self['class_accuracies'] = calculate_per_class_accuracies(y_pred, y)
        self['fpr'], self['tpr'], self['roc_auc'], self['roc_thresholds'] = calculate_auc(y_pred, y)
        self['roc_auc_avg'] = sum(self['roc_auc']) / n_classes
        self['precision'], self['recall'] = calculate_precision_recall(y_pred, y)
        self['confusion_matrix'] = calculate_confusion_matrix(y_pred, y)


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



def evaluate_classifier(
    mltk_model:MltkModel,
    tflite:bool=False,
    weights:str=None,
    max_samples_per_class:int=-1,
    classes:List[str]=None,
    verbose:bool=False,
    show:bool=False,
    update_archive:bool=True,
    **kwargs
) -> ClassifierEvaluationResults:
    """Evaluate a trained classification model
    
    Args:
        mltk_model: MltkModel instance
        tflite: If true then evalute the .tflite (i.e. quantized) model, otherwise evaluate the keras model
        weights: Optional weights to load before evaluating (only valid for a keras model)
        max_samples_per_class: Maximum number of samples per class to evaluate. This is useful for large datasets
        classes: Specific classes to evaluate
        verbose: Enable progress bar
        show: Show the evaluation results diagrams
        update_archive: Update the model archive with the eval results

    Returns:
        Dictionary containing evaluation results
    """

    if not isinstance(mltk_model, TrainMixin):
        raise Exception('MltkModel must inherit TrainMixin')
    if not isinstance(mltk_model, EvaluateClassifierMixin):
        raise Exception('MltkModel must inherit EvaluateClassifierMixin')
    if not isinstance(mltk_model, DatasetMixin):
        raise Exception('MltkModel must inherit a DatasetMixin')

    subdir = 'eval/tflite' if tflite else 'eval/h5'
    eval_dir = mltk_model.create_log_dir(subdir, delete_existing=True)
    logger = mltk_model.create_logger('eval', parent=get_mltk_logger())

    if update_archive:
        update_archive = mltk_model.check_archive_file_is_writable()
    gpu.initialize(logger=logger)


    try:
        mltk_model.load_dataset(
            subset='evaluation', 
            max_samples_per_class=max_samples_per_class,
            classes=classes,
            test=mltk_model.test_mode_enabled
        )
    except Exception as e:
        prepend_exception_msg(e, 'Failed to load model evaluation dataset')
        raise
    

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

    logger.info(mltk_model.summarize_dataset())

    y_pred = []
    y_label = []

    progbar =_get_progbar(mltk_model, verbose)
    for batch_x, batch_y in _iterate_evaluation_data(mltk_model):
        if isinstance(built_model, KerasModel):
            pred = built_model.predict(batch_x, verbose=0)
        else:
            pred = built_model.predict(batch_x, y_dtype=np.float32)

        if progbar is not None:
            progbar.update(len(pred))
        
        y_pred.extend(pred)
        if batch_y.shape[-1] == 1 or len(batch_y.shape) == 1:
            y_label.extend(batch_y)
        else:
            y_label.extend(np.argmax(batch_y, -1))

    if progbar is not None:
        progbar.close()

    mltk_model.unload_dataset()

    results = ClassifierEvaluationResults(
        name=mltk_model.name,
        classes=getattr(mltk_model, 'classes', None)
    )

    y_pred = _list_to_numpy_array(y_pred)
    y_label = np.asarray(y_label, dtype=np.int32)

    results.calculate(
        y=y_label,
        y_pred=y_pred,
    )

    eval_results_path = f'{eval_dir}/eval-results.json'
    with open(eval_results_path, 'w') as f:
        json.dump(results, f)
    logger.debug(f'Generated {eval_results_path}')

    summary_path = f'{eval_dir}/summary.txt'
    with open(summary_path, 'w') as f:
        f.write(results.generate_summary())
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

    if show:
        plt.show(block=True)

    logger.close() # Close the eval logger

    return results
    

def plot_results(
    results:ClassifierEvaluationResults, 
    show=True, 
    output_dir:str=None, 
    logger:logging.Logger=None
):
    """Use Matlibplot to generate plots of the evaluation results"""
    plot_roc(results, show=show, output_dir=output_dir, logger=logger)
    plot_precision_vs_recall(results, show=show, output_dir=output_dir, logger=logger)
    plot_tpr(results, show=show, output_dir=output_dir, logger=logger)
    plot_fpr(results, show=show, output_dir=output_dir, logger=logger)
    plot_tpr_and_fpr(results, show=show, output_dir=output_dir, logger=logger)
    plot_confusion_matrix(results, show=show, output_dir=output_dir, logger=logger)



def summarize_results(results: ClassifierEvaluationResults) -> str:
    """Generate a summary of the evaluation results"""
    classes = results['classes']
    class_accuracies = zip(classes, results['class_accuracies'])
    class_accuracies = sorted(class_accuracies, key=lambda x: x[1], reverse=True)

    class_auc = zip(classes, results['roc_auc'])
    class_auc = sorted(class_auc, key=lambda x: x[1], reverse=True)

    s = ''
    s += 'Overall accuracy: {:.3f}%\n'.format(results['overall_accuracy'] * 100)
    s += 'Class accuracies:\n'
    for class_label, acc in class_accuracies:
        s += '- {} = {:.3f}%\n'.format(class_label, acc * 100)

    s += 'Average ROC AUC: {:.3f}%\n'.format(results['roc_auc_avg'] * 100)
    s += 'Class ROC AUC:\n'
    for class_label, auc in class_auc:
        s += '- {} = {:.3f}%\n'.format(class_label, auc * 100)

    return s


def calculate_overall_accuracy(y_pred:np.ndarray, y_label:np.ndarray) -> float:
    """ Classifier overall accuracy calculation
    y_pred contains model predictions [n_samples, n_classes]
    y_label list of each correct class id per sample [n_samples]

    Return overall accuracy (i.e. ratio) as float
    """
    n_samples = len(y_pred)
    y_pred_label = np.argmax(y_pred, axis=1)
    correct = np.sum(y_label == y_pred_label)
    return correct / n_samples


def calculate_per_class_accuracies(y_pred:np.ndarray, y_label:np.ndarray) -> List[float]:
    """Classifier accuracy per class

    y_pred contains model predictions [n_samples, n_classes]
    y_label list of each correct class id per sample [n_samples]

    Return list of each classes' accuracy
    """

    n_samples, n_classes = y_pred.shape

    # Initialize array of accuracies
    accuracies = np.zeros(n_classes)

    # Loop on classes
    for class_id in range(n_classes):
        true_positives = 0
        # Loop on all predictions
        for i in range(n_samples):
            # Check if it matches the class that we are working on
            if y_label[i] == class_id:
                # Get prediction label
                y_pred_label = np.argmax(y_pred[i,:])
                # Check if the prediction is correct
                if y_pred_label == class_id:
                    true_positives += 1

        accuracies[class_id] = _safe_divide(true_positives, np.sum(y_label == class_id))

    return accuracies.tolist()


def calculate_auc(y_pred:np.ndarray, y_label:np.ndarray, threshold=.01) -> Tuple[float, float, List[float], List[float]]:
    """Classifier ROC AUC calculation

    y_pred contains model predictions [n_samples, n_classes]
    y_label list of each correct class id per sample [n_samples]
    thresholds Optional list of thresholds to consider

    Return tuple:
    false positive rate, true positive rate, list ROC AUC for each class, list of thresholds 
    """
    n_samples, n_classes = y_pred.shape
  
    # thresholds, linear range
    thresholds = np.arange(0.0, 1.01, threshold)

    n_thresholds = len(thresholds)

    # false positive rate
    fpr = np.zeros((n_classes, n_thresholds))
    # true positive rate
    tpr = np.zeros((n_classes, n_thresholds))
    # area under curve
    roc_auc = np.zeros(n_classes)

    # get number of positive and negative examples in the dataset
    for class_item in range(n_classes):
        # Sum of all true positive answers
        all_positives = sum(y_label == class_item)
        # Sum of all true negative answers
        all_negatives = len(y_label) - all_positives

        # iterate through all thresholds and determine fraction of true positives
        # and false positives found at this threshold
        for threshold_item in range(1, n_thresholds):
            threshold = thresholds[threshold_item]
            false_positives = 0
            true_positives = 0
            for i in range(n_samples):
                # Check prediction for this threshold
                if (y_pred[i, class_item] > threshold):
                    if y_label[i] == class_item:
                        true_positives += 1
                    else:
                        false_positives += 1
            fpr[class_item, threshold_item] = _safe_divide(false_positives, float(all_negatives))
            tpr[class_item, threshold_item] = _safe_divide(true_positives, float(all_positives))

            # Force boundary condition
            fpr[class_item,0] = 1
            tpr[class_item,0] = 1

        # calculate area under curve, trapezoid integration
        for threshold_item in range(len(thresholds)-1):
            roc_auc[class_item] += .5*(tpr[class_item,threshold_item]+tpr[class_item,threshold_item+1])*(fpr[class_item,threshold_item]-fpr[class_item,threshold_item+1])


    return fpr.tolist(), tpr.tolist(), roc_auc.tolist(), thresholds.tolist()


def calculate_precision_recall(y_pred:np.ndarray, y_label:np.ndarray) -> Tuple:
    """Calculate precision and recall
    """

    _, n_classes = y_pred.shape

    precision = [None] * n_classes
    recall = [None] * n_classes


    y_true = _label_binarize(y_label)

    for class_id in range(n_classes):
        class_precision, class_recall, _ = precision_recall_curve(y_true[:, class_id], y_pred[:, class_id])
        precision[class_id] = class_precision.tolist()
        recall[class_id] = class_recall.tolist()

    return precision, recall


def calculate_confusion_matrix(y_pred:np.ndarray, y_label:np.ndarray):
    """Calculate the confusion matrix

    """
    y_true = _label_binarize(y_label)

    cm_npy =  confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    return cm_npy.tolist()


def plot_roc(results:dict, output_dir:str, show:bool, logger: logging.Logger):
    """Generate a plot of the AUC ROC evaluation results"""
    name = results['name']
    classes = results['classes']
    fpr = results['fpr']
    tpr = results['tpr']
    roc_auc = results['roc_auc']
    n_classes = len(classes)

    title = f'ROC: {name}'
    fig = plt.figure(title)

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'AUC: {roc_auc[i]:0.5f} ({classes[i]})')
    plt.plot([], [], ' ', label='Average AUC: {:.5f}'.format(results['roc_auc_avg']))

    plt.xlim([0.0, 0.1])
    plt.ylim([0.5, 1.01])
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.grid(which='major')
    
    if output_dir:
        output_path = output_dir + f'/{name}-roc.png'
        plt.savefig(output_path)
        logger.debug(f'Generated {output_path}')
    if show:
        plt.show(block=False)
    else:
        fig.clear()
        plt.close(fig)


def plot_precision_vs_recall(results:dict, output_dir:str, show, logger: logging.Logger):
    """Generate a plot of the precision vs recall"""
    
    name = results['name']
    classes = results['classes']
    precision = results['precision']
    recall = results['recall']
    n_classes = len(classes)

    title = f'Precision vs Recall: {name}'
    fig = plt.figure(title)

    for i in range(n_classes):
        plt.plot(recall[i], precision[i], label=_normalize_class_name(classes[i]))
    
    plt.xlim([0.5, 1.0])
    plt.ylim([0.5, 1.01])
    plt.legend(loc="lower left")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.grid()

    if output_dir:
        output_path = output_dir + f'/{name}-precision_vs_recall.png'
        plt.savefig(output_path)
        logger.debug(f'Generated {output_path}')
    if show:
        plt.show(block=False)
    else:
        fig.clear()
        plt.close(fig)


def plot_tpr(results:dict, output_dir:str, show, logger: logging.Logger):
    """Generate a plot of the threshold vs TPR"""
    
    name = results['name']
    classes = results['classes']
    tpr = results['tpr']
    thresholds = results['roc_thresholds']
    n_classes = len(classes)

    title = f'Thres vs True Positive: {name}'
    fig = plt.figure(title)

    for i in range(n_classes):
        plt.plot(thresholds, tpr[i], label=_normalize_class_name(classes[i]))
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.8, 1.01])
    plt.legend(loc="lower left")
    plt.xlabel('Threshold')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.grid()

    if output_dir:
        output_path = output_dir + f'/{name}-tpr.png'
        plt.savefig(output_path)
        logger.debug(f'Generated {output_path}')
    if show:
        plt.show(block=False)
    else:
        fig.clear()
        plt.close(fig)


def plot_fpr(results:dict, output_dir:str, show, logger: logging.Logger):
    """Generate a plot of the threshold vs FPR"""
    
    name = results['name']
    classes = results['classes']
    fpr = results['fpr']
    thresholds = results['roc_thresholds']
    n_classes = len(classes)

    title = f'Thres vs False Positive: {name}'
    fig = plt.figure(title)

    for i in range(n_classes):
        plt.plot(thresholds, fpr[i], label=_normalize_class_name(classes[i]))
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 0.1])
    plt.legend(loc="upper right")
    plt.xlabel('Threshold')
    plt.ylabel('False Positive Rate')
    plt.title(title)
    plt.grid()

    if output_dir:
        output_path = output_dir + f'/{name}-fpr.png'
        plt.savefig(output_path)
        logger.debug(f'Generated {output_dir}')
    if show:
        plt.show(block=False)
    else:
        fig.clear()
        plt.close(fig)


def plot_tpr_and_fpr(results:dict, output_dir:str, show, logger: logging.Logger):
    """Generate a plot of the threshold vs FPR"""
    
    name = results['name']
    classes = results['classes']
    tpr = results['tpr']
    fpr = results['fpr']
    thresholds = results['roc_thresholds']
    n_classes = len(classes)

    title = f'Thres vs True/False Positive: {name}'
    fig = plt.figure(title)

    for i in range(n_classes):
        plt.plot(thresholds, fpr[i], label=f'FPR: {_normalize_class_name(classes[i])}')
        plt.plot(thresholds, tpr[i], label=f'TPR: {_normalize_class_name(classes[i])}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.legend(loc="center left", bbox_to_anchor=(1, 0))
    plt.xlabel('Threshold')
    plt.ylabel('True/False Positive Rate')
    plt.title(title)
    plt.grid()

    if output_dir:
        output_path = output_dir + f'/{name}-tfp_fpr.png'
        plt.savefig(output_path, bbox_inches='tight')
        logger.debug(f'Generated {output_dir}')
    if show:
        plt.show(block=False)
    else:
        fig.clear()
        plt.close(fig)


def plot_confusion_matrix(results:dict, output_dir:str, show, logger: logging.Logger):
    """Generate a plot of the confusion matrix"""

    name = results['name']
    classes = results['classes']
    cm = results['confusion_matrix']
    n_classes = len(classes)

    title = f'Confusion Matrix: {name}'
    fig = plt.figure(title, figsize=(6,6))
    ax = fig.subplots()

    ax.imshow(cm)
    
    # We want to show all ticks
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    # ... and label them with the respective list entries
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), 
            rotation=45, ha="right",
             rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, cm[i][j],
                    ha="center", va="center", color="w", 
                    backgroundcolor=(0.41, 0.41, 0.41, 0.25))
    
    ax.set_ylabel('Actual class')
    ax.set_xlabel('Predicted class')
    plt.title(title)
    
    if output_dir:
        output_path = output_dir + f'/{name}-confusion_matrix.png'
        plt.savefig(output_path)
        logger.debug(f'Generated {output_dir}')
    if show:
        plt.show(block=False)
    else:
        fig.clear()
        plt.close(fig)
    

def _safe_divide(num, dem):
    """Standard division but it denominator is 0 then return 0"""
    if dem == 0:
        return 0
    else:
        return num / dem


def _label_binarize(y_label):
    """This calls label_binarize() but ensures the return value 
    always has the shape: [n_samples, n_classes]"""

    # If n_classes == 2:
    #   y_true = (n_samples, 1)
    # else:
    #   y_true = (n_samples, n_classes)
    y_true = label_binarize(y=y_label, classes=np.arange(np.max(y_label)+1))

    # Handle case with only 2 classes
    if y_true.shape[1] == 1:
        y_tmp = np.empty((y_true.shape[0], 2), dtype=y_true.dtype)
        y_tmp[:, 0] = 1-y_true[:, 0]
        y_tmp[:, 1] = y_true[:, 0]
        y_true = y_tmp

    return y_true


def _normalize_class_name(label:str) -> str:
    if label.startswith('_'):
        label = label[1:]
    if label.endswith('_'):
        label = label[:-1]
    return label


def _iterate_evaluation_data(mltk_model:MltkModel):
    x = mltk_model.validation_data
    if x is None:
        x = mltk_model.x

    y = mltk_model.y

    if y is not None:
        if isinstance(x, tf.Tensor):
            x = x.numpy()
            y = y.numpy()

        if isinstance(x, np.ndarray):
            yield x, y

        else:
            for batch_x, batch_y in zip(x, y):
                batch_x = _convert_tf_tensor_to_numpy_array(batch_x, expand_dim=0)
                batch_y = _convert_tf_tensor_to_numpy_array(batch_y, expand_dim=0)
                yield batch_x, batch_y 
    
    else:
        for batch in x:
            batch_x, batch_y, _ = tf.keras.utils.unpack_x_y_sample_weight(batch)
            batch_x = _convert_tf_tensor_to_numpy_array(batch_x)
            batch_y = _convert_tf_tensor_to_numpy_array(batch_y)
            yield batch_x, batch_y
            

def _list_to_numpy_array(python_list:List[np.ndarray], dtype=None) -> np.ndarray:
    n_samples = len(python_list)
    if len(python_list[0].shape) > 0:
        numpy_array_shape = (n_samples,) + python_list[0].shape
    else:
        numpy_array_shape = (n_samples,)
    
    numpy_array = np.empty(numpy_array_shape, dtype=dtype or python_list[0].dtype)
    for i, pred in enumerate(python_list):
        numpy_array[i] = pred

    return numpy_array


def _convert_tf_tensor_to_numpy_array(x, expand_dim=None):
    if isinstance(x, tf.Tensor):
        x = x.numpy()
        
    elif isinstance(x, (list,tuple)):
        if isinstance(x[0], np.ndarray) and expand_dim is not None:
            return x
        
        retval = []
        for i in x:
            retval.append(_convert_tf_tensor_to_numpy_array(i, expand_dim=expand_dim))

        return tuple(retval)

    if expand_dim is not None:
        x = np.expand_dims(x, axis=expand_dim)
    
    return x


def _get_progbar(mltk_model:MltkModel, verbose:bool) -> tqdm.tqdm:
    if not verbose:
        return None 

    try:
        class_counts = getattr(mltk_model, 'class_counts', {})
        eval_class_counts = class_counts.get('evaluation', {})
        valid_class_counts = class_counts.get('validation', {})
        eval_n_samples = sum(eval_class_counts.values())
        valid_n_samples = sum(valid_class_counts.values())
        if eval_n_samples > 0:
            n_samples = eval_n_samples
        elif valid_n_samples > 0:
            n_samples = valid_n_samples
        else:
            n_samples = sum(class_counts.values()) or None

    except:
        n_samples = None

    return tqdm.tqdm(unit='prediction', desc='Evaluating', total=n_samples)