from __future__ import annotations
from typing import Union, Callable, Tuple
import types


import numpy as np
from mltk.utils.python import prepend_exception_msg
from .base_mixin import BaseMixin


from ..model_attributes import MltkModelAttributesDecorator, CallableType


@MltkModelAttributesDecorator()
class DatasetMixin(BaseMixin):
    """Provides generic dataset properties to the base :py:class:`~MltkModel`
    
    Refer to te `Model Specification <https://siliconlabs.github.io/mltk/docs/guides/model_specification.html>`_ guide for more details."""
    
    @property
    def x(self):
        """Input data
        
        It could be:

        - A Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs).
        - A TensorFlow tensor, or a list of tensors (in case the model has multiple inputs).
        - A dict mapping input names to the corresponding array/tensors, if the model has named inputs.
        - A tf.data dataset. Should return a tuple of either (inputs, targets) or (inputs, targets, sample_weights).
        - A generator or keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample_weights). 
          A more detailed description of unpacking behavior for iterator types (Dataset, generator, Sequence) is given below.
        """
        return self._attributes.get_value('dataset.x', default=None)
    @x.setter 
    def x(self, v):
        self._attributes['dataset.x'] = v 


    @property
    def y(self):
        """Target data
        
        Like the input data x, it could be either Numpy array(s) or TensorFlow tensor(s). 
        It should be consistent with x (you cannot have Numpy inputs and tensor targets, or inversely). 
        If x is a dataset, generator, or keras.utils.Sequence instance, y should not be specified (since targets will be obtained from x).
        """
        return self._attributes.get_value('dataset.y', default=None)
    @y.setter 
    def y(self, v):
        self._attributes['dataset.y'] = v 


    @property
    def validation_split(self) -> float:
        """Float between 0 and 1
        Fraction of the training data to be used as validation data. 
        The model will set apart this fraction of the training data, will not train on it, 
        and will evaluate the loss and any model metrics on this data at the end of each epoch. 
        The validation data is selected from the last samples in the x and y data provided, before shuffling. 
        This argument is not supported when x is a dataset, generator or keras.utils.Sequence instance.
        """
        return self._attributes.get_value('dataset.validation_split', default=0.0)
    @validation_split.setter
    def validation_split(self, v: float):
        self._attributes['dataset.validation_split'] = v 


    @property 
    def validation_data(self):
        """Data on which to evaluate the loss and any model metrics at the end of each epoch. 
        The model will not be trained on this data. Thus, note the fact that the validation loss of 
        data provided using validation_split or validation_data is not affected by regularization 
        layers like noise and dropout. validation_data will override validation_split. 
        validation_data could be:

        - tuple (x_val, y_val) of Numpy arrays or tensors
        - tuple (x_val, y_val, val_sample_weights) of Numpy arrays
        - dataset For the first two cases, batch_size must be provided. For the last case, validation_steps 
          could be provided. Note that validation_data does not support all the data types that are supported 
          in x, eg, dict, generator or keras.utils.Sequence.
        """
        return self._attributes.get_value('dataset.validation_data', default=None)
    @validation_data.setter
    def validation_data(self, v):
        self._attributes['dataset.validation_data'] = v 


    @property
    def shuffle(self) -> bool:
        """	Boolean (whether to shuffle the training data before each epoch) or str (for 'batch'). 
        This argument is ignored when x is a generator. 'batch' is a special option for dealing with the limitations of HDF5 data; 
        it shuffles in batch-sized chunks. Has no effect when steps_per_epoch is not None.
        """
        return self._attributes.get_value('dataset.shuffle', default=True)
    @shuffle.setter
    def shuffle(self, v: bool):
        self._attributes['dataset.shuffle'] = v 


    @property
    def class_weights(self) -> Union[str,dict]:
        """Specifies how class weights should be calculated.  
        Default: `None`

        This can be useful to tell the model to "pay more attention" to samples from an under-represented class.

        May be one of the following:

        - If ``balanced`` is given, class weights will be given by: ``n_samples / (n_classes * np.bincount(y))``
        - If a dictionary is given, keys are classes and values are corresponding class weights.
        - If ``None`` is given, the class weights will be uniform.
        """
        return self._attributes.get_value('dataset.class_weights', default=None)
    @class_weights.setter
    def class_weights(self, v: Union[str,dict]):
        self._attributes['dataset.class_weights'] = v


    @property
    def class_counts(self) -> dict:
        """Dictionary of samples counts for each class

        This is used for generating a summary of the dataset or when calculating class weights
        when ``my_model.class_weights=balanced``.
        
        The dictionary may contain sub-dictionaries for each subset of the dataset, e.g.:

        .. highlight:: python
        .. code-block:: python

            my_model.class_counts = dict(
                training = dict(
                    cat = 100,
                    dog = 200,
                    goat = 500
                ),
                validation = dict(
                    cat = 10,
                    dog = 20,
                    goat = 50
                ),
                evaluation = dict(
                    cat = 10,
                    dog = 20,
                    goat = 50
                )
            )

        Or it may contain just class/counts, e.g.:
    
        .. highlight:: python
        .. code-block:: python

            my_model.class_counts = dict(
                cat = 100,
                dog = 200,
                goat = 500
            )
        
        """
        return self._attributes.get_value('dataset.class_counts', default=dict(
            training={},
            validation={},
            evaluation={}
        ))
    @class_counts.setter
    def class_counts(self, v: dict):
        self._attributes['dataset.class_counts'] = v


    @property
    def sample_weight(self) -> np.ndarray:
        """Optional Numpy array of weights for the training samples, 
        used for weighting the loss function (during training only). 
        You can either pass a flat (1D) Numpy array with the same length as the input samples 
        (1:1 mapping between weights and samples), or in the case of temporal data, you can pass 
        a 2D array with shape (samples, sequence_length), to apply a different weight to every
        timestep of every sample. This argument is not supported when x is a dataset, generator,
        or keras.utils.Sequence instance, instead provide the sample_weights as the third element of x.
        """
        return self._attributes.get_value('dataset.sample_weight', default=None)
    @sample_weight.setter
    def sample_weight(self, v: np.ndarray):
        self._attributes['dataset.sample_weight'] = v


    @property
    def steps_per_epoch(self) -> int:
        """Integer or None. Total number of steps (batches of samples) before 
        declaring one epoch finished and starting the next epoch. 
        When training with input tensors such as TensorFlow data tensors, the default None 
        is equal to the number of samples in your dataset divided by the batch size, 
        or 1 if that cannot be determined. If x is a tf.data dataset, and 'steps_per_epoch' 
        is None, the epoch will run until the input dataset is exhausted. 
        When passing an infinitely repeating dataset, you must specify the steps_per_epoch argument. 
        This argument is not supported with array inputs.
        """
        return self._attributes.get_value('dataset.steps_per_epoch', default=None)
    @steps_per_epoch.setter 
    def steps_per_epoch(self, v: int):
        self._attributes['dataset.steps_per_epoch'] = v


    @property
    def validation_steps(self) -> int:
        """Only relevant if validation_data is provided and is a tf.data dataset. 
        Total number of steps (batches of samples) to draw before stopping when 
        performing validation at the end of every epoch. If 'validation_steps' is None, 
        validation will run until the validation_data dataset is exhausted. 
        In the case of an infinitely repeated dataset, it will run into an infinite loop. 
        If 'validation_steps' is specified and only part of the dataset will be consumed, 
        the evaluation will start from the beginning of the dataset at each epoch. 
        This ensures that the same validation samples are used every time.
        """
        return self._attributes.get_value('dataset.validation_steps', default=None)
    @validation_steps.setter 
    def validation_steps(self, v: int):
        self._attributes['dataset.validation_steps'] = v


    @property
    def validation_batch_size(self) -> int:
        """Integer or None. Number of samples per validation batch. 
        If unspecified, will default to batch_size. Do not specify 
        the validation_batch_size if your data is in the form of datasets, generators, 
        or keras.utils.Sequence instances (since they generate batches).
        """
        return self._attributes.get_value('dataset.validation_batch_size', default=None)
    @validation_batch_size.setter 
    def validation_batch_size(self, v: int):
        self._attributes['dataset.validation_batch_size'] = v


    @property
    def validation_freq(self) -> int:
        """Only relevant if validation data is provided. 
        Integer or collections_abc.Container instance (e.g. list, tuple, etc.). 
        If an integer, specifies how many training epochs to run before a new validation run is performed, 
        e.g. validation_freq=2 runs validation every 2 epochs. If a Container, specifies the epochs on which to run validation, 
        e.g. validation_freq=[1, 2, 10] runs validation at the end of the 1st, 2nd, and 10th epochs.
        """
        return self._attributes.get_value('dataset.validation_freq', default=1)
    @validation_freq.setter 
    def validation_freq(self, v: int):
        self._attributes['dataset.validation_freq'] = v


    @property
    def loaded_subset(self) -> str:
        """The currently loaded dataset subset: training, validation, evaluation"""
        return self._attributes.get_value('dataset.loaded_subset', default=None)
    @loaded_subset.setter
    def loaded_subset(self, subset:str):
        if subset is not None:
            subset = subset.lower()
            if subset not in ('training', 'validation', 'evaluation'):
                raise Exception('Dataset subset must be one of: training, validation, evaluation')

        self._attributes['dataset.loaded_subset'] = subset


    @property
    def dataset(self) -> Union[types.ModuleType,Callable,str,MltkDataset]:
        """The model dataset

        The value of this is dependent on the implementation of this model's :py:meth:`~load_dataset` method.

        By default, this may be one of the following:

        - None, in this case :py:meth:`~load_dataset` method should be overridden
        - A reference to :py:class:`mltk.core.MltkDataset` instance
        - A reference to a function with the signature:

        .. highlight:: python
        .. code-block:: python

            def my_dataset_loader(subset: str, test:bool, **kwargs):
                # Arguments:
                #    subset: The dataset subset: training, validation or evaluation
                #    test: If true then only load a few samples for testing 
                # Returns, one of the following:
                #    - None, in this case the function is expected to set the my_model.x, my_model.y, and/or my_model.validation_data properties
                #    - x - The x samples, in this case the my_model.x properties will be automatically set
                #    - (x, y) - The x samples and y labels. In this case the my_model.x and my_model.y properties will be automatically set
                #    - (x, None, validation_data) - The x samples and validation_data. In this case the my_model.x and my_model.validation_data properties will be automatically set
                ...
        
        - A reference to a Python class instance, the class should have a method named "load_dataset" that has a similar signature as the function above
        - A reference to a Python module, the module should have a function named "load_dataset" that has a similar signature as the function above

        If this model's :py:meth:`~load_dataset` API is overridden then it may have a different value.
        For instance, the :py:class:`mltk.core.AudioDatasetMixin` and :py:class:`mltk.core.ImageDatasetMixin` override this property. 
        Refer to their documentation for more details.
        """
        return self._attributes.get_value('dataset.dataset', default=None)
    @dataset.setter
    def dataset(self, v: Union[types.ModuleType,Callable,str,MltkDataset]):
        self._attributes['dataset.dataset'] = v


    def load_dataset(
        self, 
        subset: str,  
        test:bool = False,
        **kwargs
    ):
        """Load the dataset

        This is automatically invoked by the MLTK during the train, quantize and evaluation operations.
        It loads the model dataset and configures the model properties:

        - my_model.loaded_subset
        - my_model.x
        - my_model.y
        - my_model.validation_data

        By default, this does the following:

        1. Sets the :py:attr:`~loaded_subset` property
        2. Clears the :py:attr:`~x`, :py:attr:`~y`, :py:attr:`~validation_data` properties
        3. Calls :py:attr:`~dataset`
        4. Sets the :py:attr:`~x`, :py:attr:`~y` and/or :py:attr:`~validation_data` properties based on the return value of :py:attr:`~dataset`
    
        The API may be overridden by:

        - Your MltkModel class definition
        - Another MLTK model mixin such as :py:class:`mltk.core.AudioDatasetMixin` or :py:class:`mltk.core.ImageDatasetMixin`

        Arguments:
            subset: The dataset subset: training, validation or evaluation
            test: If true then only load a few samples for testing 
        
        Returns:
            One of the following:

            - None, in this case the caller is expected to set the my_model.x, my_model.y, and/or my_model.validation_data properties
            - x - The x samples. In this case this API will automatically set the my_model.x properties
            - (x, y) - The x samples and y labels. In this case this API will automatically set the my_model.x and my_model.y properties
            - (x, None, validation_data) - The x samples and validation_data. In this case this API will automatically set the my_model.x and my_model.validation_data properties
        """
        self.loaded_subset = subset 
        dataset = self.dataset

        self.x = None 
        self.validation_data = None 
        self.y = None

        if dataset is None:
            return 

        retval = None

        if isinstance(dataset, MltkDataset):
            retval = dataset.load_dataset(subset=subset, test=test, mltk_model=self, **kwargs)

        elif callable(dataset):
            retval = dataset(subset=subset, test=test, mltk_model=self, **kwargs)

        elif isinstance(dataset, (types.ModuleType, object)):
            if not hasattr(dataset, 'load_dataset'):
                raise Exception('If a module or class is set in mltk_model.dataset, the the module/class must specify the function: load_dataset(subset:str, test:bool, **kwargs)')
        
            try:
                retval = dataset.load_dataset(subset=subset, test=test, mltk_model=self, **kwargs)
            except Exception as e:
                prepend_exception_msg(e, f'Exception while invoking mltk_model.dataset.load_dataset(): {dataset}')
                raise
    
        else:
            raise RuntimeError(
                'mltk_model.dataset must either be one of:\n'
                '- MltkDataset instance\n'
                '- callback function: load_dataset(subset:str, test:bool, **kwargs)\n'
                '- object containing method load_dataset(subset:str, test:bool, **kwargs)\n'
                '- module with function: load_dataset(subset:str, test:bool, **kwargs)\n'
                'Alternatively, ensure you model inherits the correct mixin, e.g. AudioDatasetMixin, ImageDatasetMixin'
            )

        if retval is not None:
            if isinstance(retval, str):
                raise ValueError(
                    'The value returned by the my_model.dataset callback should be one of:\n'
                    '- None, in this case the caller is expected to set the my_model.x, my_model.y, and/or my_model.validation_data properties\n'
                    '- x - The x samples. In this case this API will automatically set the my_model.x properties\n'
                    '- (x, y) - The x samples and y labels. In this case this API will automatically set the my_model.x and my_model.y properties\n'
                    '- (x, None, validation_data) - The x samples and validation_data. In this case this API will automatically set the my_model.x and my_model.validation_data properties\n'
                    'Alternatively, ensure you model inherits the correct mixin, e.g. AudioDatasetMixin, ImageDatasetMixin'
                )

            if isinstance(retval, tuple):
                if len(retval) > 3:
                    raise ValueError(
                        'The value returned by the my_model.dataset callback should be one of:\n'
                        '- None, in this case the caller is expected to set the my_model.x, my_model.y, and/or my_model.validation_data properties\n'
                        '- x - The x samples. In this case this API will automatically set the my_model.x properties\n'
                        '- (x, y) - The x samples and y labels. In this case this API will automatically set the my_model.x and my_model.y properties\n'
                        '- (x, None, validation_data) - The x samples and validation_data. In this case this API will automatically set the my_model.x and my_model.validation_data properties\n'
                        'Alternatively, ensure you model inherits the correct mixin, e.g. AudioDatasetMixin, ImageDatasetMixin'
                    )
                
                self.x = retval[0]
                if len(retval) > 1:
                    self.y = retval[1]
                if len(retval) > 2:
                    self.validation_data = retval[2]
            else:
                self.x = retval


    def unload_dataset(self):
        """Unload the dataset"""
        self.loaded_subset = None
        dataset = self.dataset

        if dataset is None or callable(dataset):
            return 

        if isinstance(dataset, MltkDataset):
            dataset.unload_dataset()
            return 
  

        if isinstance(dataset, (types.ModuleType, object)) and hasattr(dataset, 'unload_dataset'):
            try:
                return dataset.unload_dataset()
            except Exception as e:
                prepend_exception_msg(e, f'Exception while invoking mltk_model.dataset.unload_dataset(): {dataset}')
                raise
        

    def summarize_dataset(self) -> str: 
        """Summarize the dataset
        
        If my_model.dataset is provided then this attempts to call my_model.dataset.summarize_dataset().
        If my_model.dataset is not provided or does not have the summarize_dataset() method,
        then this attempts to generate a summary from my_model.class_counts.
        """
        summary = ''
        dataset = self.dataset

        if dataset is None or callable(dataset):
            pass 

        elif isinstance(dataset, MltkDataset):
            summary = dataset.summarize_dataset()

        elif isinstance(dataset, (types.ModuleType, object)) and hasattr(dataset, 'summarize_dataset'):
            try:
                summary = dataset.summarize_dataset()
            except Exception as e:
                prepend_exception_msg(e, f'Exception while invoking mltk_model.dataset.summarize_dataset(): {dataset}')
                raise

        class_counts = self.class_counts
        if not summary and class_counts:
            summary += MltkDataset.summarize_class_counts(class_counts)
        
        return summary



    def _register_attributes(self):
        self._attributes.register('dataset.dataset', dtype=(types.ModuleType,str,CallableType,object,MltkDataset))
        self._attributes.register('dataset.loaded_subset', dtype=str)
        self._attributes.register('dataset.x')
        self._attributes.register('dataset.y')
        self._attributes.register('dataset.validation_split', dtype=float)
        self._attributes.register('dataset.validation_data')
        self._attributes.register('dataset.shuffle', dtype=bool)
        self._attributes.register('dataset.class_weights', dtype=(str,dict,list))
        self._attributes.register('dataset.class_counts', dtype=dict)
        self._attributes.register('dataset.sample_weight', dtype=np.ndarray)
        self._attributes.register('dataset.steps_per_epoch', dtype=int)
        self._attributes.register('dataset.validation_steps', dtype=int)
        self._attributes.register('dataset.validation_batch_size', dtype=int)
        self._attributes.register('dataset.validation_freq', dtype=int)


        
class MltkDataset:
    """Helper class for configuring a training dataset
    
    Refer to :py:class:`mltk.core.DatasetMixin` for more details.
    """
    def load_dataset(
        self, 
        subset: str,  
        test:bool = False,
        **kwargs
    ):
        """Load the dataset subset
        
        This is called automatically by the MLTK before training
        or evaluation.
        
        Args:
            subset: The dataset subset to return: 'training' or 'evaluation'
            test: This is optional, it is used when invoking a training "dryrun", e.g.: mltk train basic_example-test
                If this is true, then only return a small portion of the dataset for testing purposes

        Return:
            One of the following:

            - None, in this case the function is expected to set the my_model.x, my_model.y, and/or my_model.validation_data properties
            - x - The x samples, in this case the my_model.x properties will be automatically set
            - (x, y) - The x samples and y labels. In this case the my_model.x and my_model.y properties will be automatically set
            - (x, None, validation_data) - The x samples and validation_data. In this case the my_model.x and my_model.validation_data properties will be automatically set
        """
        pass

    def unload_dataset(self):
        """Unload the dataset"""
        self.loaded_subset = None


    def summarize_dataset(self) -> str: 
        """Return a string summary of the dataset"""
        return ''


    @staticmethod
    def summarize_class_counts(class_counts:dict) -> str:
        """Generate a text summary of the given class counts dictionary"""
        summary = ''
        if 'training' in class_counts:
            for subset, counts in class_counts.items():
                n_samples = sum(counts.values())
                if n_samples == 0:
                    continue
                max_class_name_len = max([len(x) for x in counts.keys()])
                summary += f'Dataset subset: {subset}, found {n_samples} samples:\n'
                for key, value in counts.items():
                    summary += f'  {key.rjust(max_class_name_len)}: {value}\n'
        else:
            n_samples = sum(class_counts.values())
            max_class_name_len = max([len(x) for x in counts.keys()])
            summary += f'Dataset found {n_samples} samples:\n'
            for key, value in class_counts.items():
                summary += f'  {key.rjust(max_class_name_len)}: {value}\n'

        return summary