
import types
from typing import List, Tuple, Union, Dict, Callable

import numpy as np


from mltk.utils.python import prepend_exception_msg
from mltk.utils.process_pool_manager import ProcessPoolManager
from mltk.core.utils import (convert_y_to_labels, get_mltk_logger)

from .data_generator_dataset_mixin import (DataGeneratorDatasetMixin, DataGeneratorContext)
from ..model_attributes import MltkModelAttributesDecorator, CallableType



@MltkModelAttributesDecorator()
class ImageDatasetMixin(DataGeneratorDatasetMixin):
    """Provides image dataset properties to the base :py:class:`~MltkModel`"""

    @property
    def dataset(self) -> Union[types.ModuleType,Callable,str]:
        """Path to the image dataset's python module, a function 
        that manually loads the dataset, or the file path to a directory of samples.

        If a Python module is provided, it must implement the function:

        .. highlight:: python
        .. code-block:: python

           def load_data():
              ...
        
        The load_data() function should either return a tuple as:
        (x_train, y_train), (x_test, y_test)
        OR it should return the path to a directory containing the dataset's samples.

        If a function is provided, the function should return the tuple:
        (x_train, y_train), (x_test, y_test)
        OR it should return the path to a directory containing the dataset's samples.

        """
        return self._attributes.get_value('dataset.dataset', default=None)
    @dataset.setter
    def dataset(self, v: Union[types.ModuleType,Callable,str]):
        self._attributes['dataset.dataset'] = v

    @property
    def follow_links(self) -> bool:
        """Whether to follow symlinks inside class sub-directories 

        Default: ``True``
        """
        return self._attributes.get_value('image.follow_links', default=True)
    @follow_links.setter
    def follow_links(self, v: bool):
        self._attributes['image.follow_links'] = v


    @property
    def shuffle_dataset_enabled(self) -> bool:
        """Shuffle the dataset directory once

        Default: ``false``
        
        - If true, the dataset directory will be shuffled the first time it is processed and
            and an index containing the shuffled file names is generated in the training log directory. 
            The index is reused to maintain the shuffled order for subsequent processing.
        - If false, then the dataset samples are sorted alphabetically and saved to an index in the dataset directory. 
            The alphabetical index file is used for subsequent processing.
        
        """
        return self._attributes.get_value('image.shuffle_dataset_enabled', default=False)
    @shuffle_dataset_enabled.setter
    def shuffle_dataset_enabled(self, v: bool):
        self._attributes['image.shuffle_dataset_enabled'] = v

    @property 
    def image_classes(self) -> List[str]:
        """Return a list of class labels the model should classify"""
        return self._attributes['image.classes']
    @image_classes.setter 
    def image_classes(self, v: List[str]):
        self._attributes['image.classes'] = v


    @property 
    def image_input_shape(self) -> Tuple[int]:
        """Return the image input shape as a tuple of integers"""
        return self._attributes['image.input_shape']
    @image_input_shape.setter 
    def image_input_shape(self, v: Tuple[int]):
        self._attributes['image.input_shape'] = v


    @property 
    def target_size(self) -> Tuple[int]:
        """Return the target size of the generated images. 
        The image data generator will automatically resize all images to this size.
        If omitted, ``my_model.input_shape`` is used.

        .. note:: This is only used if providing a directory image dataset
        """
        return self._attributes.get_value('image.target_size', default=None)
    @target_size.setter 
    def target_size(self, v: Tuple[int]):
        self._attributes['image.target_size'] = v


    @property
    def class_mode(self) -> str:
        """Determines the type of label arrays that are returned.  
        Default: `categorical`

        - **categorical** -  2D one-hot encoded labels
        - **binary** - 1D binary labels
        - **sparse** - 1D integer labels
        - **input** - images identical to input images (mainly used to work with autoencoders)

        """
        return self._attributes.get_value('image.class_mode', default='categorical')
    @class_mode.setter 
    def class_mode(self, v: str):
        self._attributes['image.class_mode'] = v


    @property
    def color_mode(self) -> str:
        """The type of image data to use

        Default: ``auto``

        May be one of the following:

        - **auto** - Automatically determine the color mode based on the input shape channels
        - **grayscale** - Convert the images to grayscale (if necessary). The put shape must only have 1 channel
        - **rgb** -  The input shape must only have 3 channels
        - **rgba** - The input shape must have 4 channels

        """
        return self._attributes.get_value('image.color_mode', default='auto')
    @color_mode.setter
    def color_mode(self, v: str):
        self._attributes['image.color_mode'] = v


    @property
    def interpolation(self) -> str:
        """Interpolation method used to resample the image if the target size is different from that of the loaded image
         
        Default: ``bilinear``

        Supported methods are ``none``, ``nearest``, ``bilinear``, ``bicubic``, ``lanczos``, ``box`` and ``hamming`` .
        If ``none`` is used then the generated images are **not automatically resized**. 
        In this case, the :py:class:`mltk.core.preprocess.image.parallel_generator.ParallelImageDataGenerator` ``preprocessing_function`` argument should be used to reshape the
        image to the expected model input shape.
        """
        return self._attributes.get_value('image.interpolation', default='bilinear')
    @interpolation.setter
    def interpolation(self, v: str):
        self._attributes['image.interpolation'] = v


    @property
    def datagen(self):
        """Training data generator. 
        
        Should be a reference to a :py:class:`mltk.core.preprocess.image.parallel_generator.ParallelImageDataGenerator` instance
        OR `tensorflow.keras.preprocessing.image.ImageDataGenerator <https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator>`_
        """
        return self._attributes.get_value('image.datagen', default=None) 
    @datagen.setter
    def datagen(self, v):
        self._attributes['image.datagen'] = v


    @property
    def validation_datagen(self):
        """Validation/evaluation data generator. 

        If omitted, then ``datagen`` is used for validation and evaluation.
        
        Should be a reference to a :py:class:`mltk.core.preprocess.image.parallel_generator.ParallelImageDataGenerator` instance
        OR `tensorflow.keras.preprocessing.image.ImageDataGenerator <https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator>`_
        """
        return self._attributes.get_value('image.validation_datagen', default=None) 
    @validation_datagen.setter
    def validation_datagen(self, v):
        self._attributes['image.validation_datagen'] = v



    def load_dataset(
        self, 
        subset: str, 
        classes: List[str]=None,
        max_samples_per_class: int=-1,
        test:bool = False,
        **kwargs
    ): # pylint: disable=arguments-differ
        """Pre-process the dataset and prepare the model dataset attributes"""
        self.loaded_subset = subset

        logger = get_mltk_logger()
        ProcessPoolManager.set_logger(logger)

        # First download the dataset if necessary
        if self.dataset is None:
            raise Exception('Must specify dataset, e.g.: mltk_model.dataset = tf.keras.datasets.cifar10')
        dataset_data = load_dataset(self.dataset)

        if not classes:
            if not self.classes or not isinstance(self.classes, (list,tuple)):
                raise Exception('Must specify mltk_model.classes which must be a list of class labels')
            classes = self.classes

        if self.input_shape is None or len(self.input_shape) != 3:
            raise Exception('Must specify mltk_model.input_shape which must be a tuple (height, width, depth)')
        if self.datagen is None:
            raise Exception('Must specify mltk_model.datagen')

        if not hasattr(self, 'batch_size'):
            logger.warning('MltkModel does not define batch_size, defaulting to 32')
            batch_size = 32
        else:
            batch_size = self.batch_size

        input_depth = self.input_shape[2]
        color_mode = self.color_mode
        if color_mode == 'auto':
            if input_depth == 1:
                color_mode = 'grayscale'
            elif input_depth == 3:
                color_mode = 'rgb'
            else:
                raise Exception('mltk_model.input_shape[2] must be 1 or 3 (i.e. grayscale or rgb)')
            
        if input_depth == 1 and color_mode != 'grayscale':
            logger.warning('mltk_model.input_shape[2]=1 but mltk_model.color_mode != grayscale')
        if input_depth == 3 and color_mode != 'rgb':
            logger.warning('mltk_model.input_shape[2]=3 but mltk_model.color_mode != rgb')

        target_size = self.target_size or self.input_shape[:2]
        logger.debug(f'Target image size={target_size}')

        eval_shuffle = False
        eval_augmentation_enabled = False 
       
        if test:
            batch_size = 3
            max_samples_per_class = batch_size
            if hasattr(self, 'batch_size'):
                self.batch_size = batch_size
            self.datagen.max_batches_pending = 1
            logger.debug(f'Test mode enabled, forcing max_samples_per_class={max_samples_per_class}, batch_size={batch_size}')
            
        if self.loaded_subset == 'evaluation':
            if hasattr(self, 'eval_shuffle'):
                eval_shuffle = self.eval_shuffle
            if hasattr(self, 'eval_augment'):
                eval_augmentation_enabled = self.eval_augment
            if max_samples_per_class == -1 and hasattr(self, 'eval_max_samples_per_class'):
                max_samples_per_class = self.eval_max_samples_per_class


        train_datagen = None 
        validation_datagen = None

        if self.loaded_subset == 'training':
            training_datagen_creator = self.get_datagen_creator('training')

        # Get the validation data generator if one was specified
        # otherwise fallback to the training data generator
        validation_datagen_creator = self.get_datagen_creator('validation')


        # If a custom loading function was specified
        if isinstance(dataset_data, (tuple,list)):
            if not(len(dataset_data) == 2 or len(dataset_data) == 4):
                raise Exception('mltk_model.dataset should return a tuple of the form: (x_train, y_train), (x_test, y_test)')
            
            if len(dataset_data) == 2:
                train, test = dataset_data
                if not isinstance(train, (list, tuple)) or len(train) != 2:
                    raise Exception('mltk_model.dataset should return a tuple of the form: (x_train, y_train), (x_test, y_test)')
                if not isinstance(test, (list, tuple)) or len(test) != 2:
                    raise Exception('mltk_model.dataset should return a tuple of the form: (x_train, y_train), (x_test, y_test)')

                x_train, y_train = train
                x_test, y_test = test

            else:
                x_train, y_train, x_test, y_test = dataset_data

            if self.class_mode == 'categorical' and y_train.shape[-1] != len(classes):
                raise Exception(f'y_train.shape[-1] ({y_train.shape[-1]}) != len(mltk_model.classes) ({len(classes)}). ' \
                    'Perhaps you need to convert your dataset to categorical?')
            if self.class_mode == 'categorical' and y_test.shape[-1] != len(classes):
                raise Exception(f'y_test.shape[-1] ({y_train.shape[-1]}) != len(mltk_model.classes) ({len(classes)}). ' \
                    'Perhaps you need to convert your dataset to categorical?')


            if self.loaded_subset == 'training':
                if max_samples_per_class != -1 and self.class_mode == 'categorical':
                    x_train, y_train = _clamp_max_samples_per_class(x_train, y_train, max_samples_per_class)

                train_datagen = training_datagen_creator.flow(
                    x_train, 
                    y_train,
                    batch_size=batch_size,
                    shuffle=True
                )
                self.class_counts['training'] = _get_class_counts(train_datagen.y, classes=classes, class_mode=self.class_mode)

            if max_samples_per_class != -1 and self.class_mode == 'categorical':
                x_test, y_test = _clamp_max_samples_per_class(x_test, y_test, max_samples_per_class)

            validation_datagen = validation_datagen_creator.flow(
                x_test, 
                y_test,
                batch_size=batch_size,
                shuffle=eval_shuffle if self.loaded_subset == 'evaluation' else True
            )
            self.class_counts['validation'] = _get_class_counts(validation_datagen.y, classes=classes, class_mode=self.class_mode)

        # If a directory was specified
        elif isinstance(dataset_data, str):
            from mltk.core.preprocess.image.parallel_generator import ParallelImageDataGenerator

            shuffle_index_dir = None
            if self.shuffle_dataset_enabled:
                shuffle_index_dir = self.get_shuffle_index_dir()
                logger.debug(f'shuffle_index_dir={shuffle_index_dir}')
            
            logger.debug(f'Dataset directory: {dataset_data}')
            batch_shape = (batch_size,) + tuple(self.input_shape)
            logger.debug(f'Batch shape: {batch_shape}')
            kwargs = dict(
                directory=dataset_data,
                target_size=target_size,
                batch_shape=batch_shape,
                classes=classes,
                class_mode=self.class_mode,
                color_mode=color_mode,
                interpolation=self.interpolation,
                follow_links=self.follow_links,
                shuffle_index_dir=shuffle_index_dir,
                list_valid_filenames_in_directory_function=_get_list_valid_filenames_function(self.dataset),
            )

            if self.loaded_subset == 'training':
                training_datagen_creator.max_samples_per_class = max_samples_per_class
                if isinstance(training_datagen_creator, ParallelImageDataGenerator):
                    kwargs['class_counts'] = self.class_counts['training']

                train_datagen = training_datagen_creator.flow_from_directory(
                    subset='training',
                    shuffle=True,
                    **kwargs
                )
                kwargs.pop('class_counts', None)

            if self.loaded_subset in ('training', 'validation'):
                validation_datagen_creator.max_samples_per_class = max_samples_per_class
                if isinstance(validation_datagen_creator, ParallelImageDataGenerator):
                    kwargs['class_counts'] = self.class_counts['validation']
                
                validation_datagen = validation_datagen_creator.flow_from_directory(
                    subset='validation',
                    shuffle=True,
                    **kwargs
                )
                kwargs.pop('class_counts', None)

            if self.loaded_subset == 'evaluation':
                validation_datagen_creator.max_samples_per_class = max_samples_per_class
                validation_datagen_creator.validation_augmentation_enabled = eval_augmentation_enabled

                if isinstance(validation_datagen_creator, ParallelImageDataGenerator):
                    kwargs['class_counts'] = self.class_counts['validation']
                validation_datagen = validation_datagen_creator.flow_from_directory(
                    subset='validation',
                    shuffle=eval_shuffle,
                    **kwargs
                )
                kwargs.pop('class_counts', None)
         
        else:
            raise Exception(
                'mltk_model.dataset must return return a tuple as: (x_train, y_train), (x_test, y_test)'
                ' or a file path to a directory of samples'
            )


        # Fix issue with:
        # tensorflow.keras.preprocessing.image.ImageDataGenerator 
        _patch_image_iterator(validation_datagen)
        if self.class_counts['validation']:
            validation_datagen.max_samples = sum(self.class_counts['validation'].values())

        self.x = None
        self.validation_data = None

        if self.loaded_subset == 'training':
            self.x = train_datagen

        if self.loaded_subset in ('training', 'validation'):
            self.validation_data = validation_datagen
        
        if self.loaded_subset == 'evaluation':
            self.x = train_datagen if validation_datagen is None else validation_datagen


        self.datagen_context = DataGeneratorContext(
            subset = self.loaded_subset,
            train_datagen = train_datagen,
            train_class_counts = self.class_counts['training'],
            validation_datagen = validation_datagen,
            validation_class_counts = self.class_counts['validation']
        )




    def _register_attributes(self):
        from mltk.core.preprocess.image.parallel_generator import ParallelImageDataGenerator
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        self._attributes.register('image.follow_links', dtype=bool)
        self._attributes.register('image.shuffle_dataset_enabled', dtype=bool)
        self._attributes.register('image.input_shape', dtype=(list,tuple))
        self._attributes.register('image.target_size', dtype=(list,tuple))
        self._attributes.register('image.classes', dtype=(list,tuple))
        self._attributes.register('image.class_mode', dtype=str)
        self._attributes.register('image.color_mode', dtype=str)
        self._attributes.register('image.interpolation', dtype=str)
        self._attributes.register('image.datagen', dtype=(ParallelImageDataGenerator, ImageDataGenerator))
        self._attributes.register('image.validation_datagen', dtype=(ParallelImageDataGenerator, ImageDataGenerator))

        # We cannot call attributes while we're registering them
        # So we return a function that will be called after
        # all the attributes are registered
        def register_parameters_populator():
            self.add_model_parameter_populate_callback(self._populate_image_dataset_model_parameters)
        
        return register_parameters_populator


    def _populate_image_dataset_model_parameters(self):
        """Populate the image processing parameters required at inference time

        These parameters will be added to the compiled .tflite TfliteModelParameters metadata.
        At inference time, these paramaters are retrieved from the generated .tflite and 
        used them to process the input images.

        NOTE: This is invoked during the compile_model() API execution.
        """
        if self.datagen is not None:
            self.set_model_parameter('samplewise_norm.rescale', float(self.datagen.rescale or 0.))
            self.set_model_parameter('samplewise_norm.mean_and_std', self.datagen.samplewise_center and self.datagen.samplewise_std_normalization)
    

def load_dataset(dataset) -> Union[str,tuple]:
    if isinstance(dataset,str):
        return dataset 

    if callable(dataset):
        try:
            return dataset()
        except Exception as e:
            prepend_exception_msg(e, f'Exception while invoking mltk_model.dataset function: {dataset}')
            raise
    
    if isinstance(dataset, (types.ModuleType, object)):
        if not hasattr(dataset, 'load_data'):
            raise Exception('If a module or class is set in mltk_model.dataset, the the module/class must specify the function: load_data()')
       
        try:
            return dataset.load_data()
        except Exception as e:
            prepend_exception_msg(e, f'Exception while invoking mltk_model.dataset.load_data(): {dataset}')
            raise
    
    raise Exception('mltk_model.dataset must either be file path to a dictionary or callback function')


def _get_list_valid_filenames_function(dataset):
    if isinstance(dataset, (types.ModuleType, object)):
        if hasattr(dataset, 'list_valid_filenames_in_directory'):
            return getattr(dataset, 'list_valid_filenames_in_directory')

    return None


def _patch_image_iterator(datagen):
    """Patch the KerasImageIterator so that 
    tensorflow.keras.preprocessing.image.ImageDataGenerator 
    properly iterates while predicting
    """
    from mltk.core.keras import ImageIterator
    if not isinstance(datagen, ImageIterator):
        return


    datagen.max_samples = -1
    datagen.sample_count = 0

    def _patched_next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """

        if datagen.max_samples > 0 and datagen.sample_count > datagen.max_samples:
            raise StopIteration()

        with self.lock:
            index_array = next(self.index_generator)

        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_samples = self._get_batches_of_transformed_samples(index_array) # pylint: disable=protected-access
        datagen.sample_count += len(batch_samples[0])
        return batch_samples

    datagen.next = types.MethodType(_patched_next, datagen)


def _clamp_max_samples_per_class(x, y, max_samples_per_class):
    """Clamp the number of samples per class to a maximum if necessary"""
    labels = convert_y_to_labels(y)
    _, class_max_counts = np.unique(labels, return_counts=True)

    n_samples = 0
    for c in class_max_counts:
        n_samples += min(c, max_samples_per_class)

    x_truncated = np.empty((n_samples, *x.shape[1:]), dtype=x.dtype)
    y_truncated = np.empty((n_samples, *y.shape[1:]), dtype=y.dtype)

    class_counts = np.zeros((len(class_max_counts),), dtype=np.int32)
    index = 0
    for i, class_id in enumerate(labels):
        if index == n_samples:
            break
        if class_counts[class_id] == max_samples_per_class:
            continue 
        class_counts[class_id] += 1
       
        x_truncated[index, :] = x[i]
        y_truncated[index, :] = y[i]
        index += 1

    return x_truncated, y_truncated


def _get_class_counts(y, classes:List[str], class_mode:str) -> Dict[str,int]:
    class_counts = {}

    if class_mode == 'categorical':
       y = convert_y_to_labels(y)
   
    if class_mode != 'input':
        for i, class_name in enumerate(classes):
            class_counts[class_name] = 0

        counts = np.bincount(y)
        for i, count in enumerate(counts):
            class_name = classes[i]
            class_counts[class_name] = count

    return class_counts