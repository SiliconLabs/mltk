from typing import Union

import numpy as np
from .base_mixin import BaseMixin


from ..model_attributes import MltkModelAttributesDecorator



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


    def load_dataset(
        self, 
        subset: str,  
        test:bool = False,
        **kwargs
    ):
        """Load the dataset

        .. note:: 
           By default this API does not do anything. 
           It should be overridden by a parent class.

        Arguments:
            subset: The dataset subset: training, validation or evaluation
            test: If true then only load a few samples for testing 
        """
        subset = subset.lower()
        if subset not in ('training', 'validation', 'evaluation'):
            raise Exception('Dataset subset must be one of: training, validation, evaluation')

        self._attributes['dataset.loaded_subset'] = subset


    def unload_dataset(self):
        """Unload the dataset"""
        self._attributes['dataset.loaded_subset'] = None


    def summarize_dataset(self) -> str: 
        return ''



    def _register_attributes(self):
        self._attributes.register('dataset.loaded_subset', dtype=str)
        self._attributes.register('dataset.x')
        self._attributes.register('dataset.y')
        self._attributes.register('dataset.validation_split', dtype=float)
        self._attributes.register('dataset.validation_data')
        self._attributes.register('dataset.shuffle', dtype=bool)
        self._attributes.register('dataset.class_weights', dtype=(str,dict,list))
        self._attributes.register('dataset.sample_weight', dtype=np.ndarray)
        self._attributes.register('dataset.steps_per_epoch', dtype=int)
        self._attributes.register('dataset.validation_steps', dtype=int)
        self._attributes.register('dataset.validation_batch_size', dtype=int)
        self._attributes.register('dataset.validation_freq', dtype=int)


        
       