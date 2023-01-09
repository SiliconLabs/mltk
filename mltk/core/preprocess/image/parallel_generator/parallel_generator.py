
import random
import copy
import os
from typing import Dict

import numpy as np
from PIL import ImageEnhance

from mltk.core.keras import (
    ImageDataGenerator,
    array_to_img, 
    img_to_array
)


from .directory_iterator import ParallelDirectoryIterator 



class ParallelImageDataGenerator(ImageDataGenerator):
    '''Generate batches of tensor image data with real-time data augmentation.

    The data will be looped over (in batches).

    This class works the exact same as the `Keras ImageDataGenerator <https://keras.io/preprocessing/image>`_
    except images are processed in the background using the standard Python 
    `multiprocessing <https://docs.python.org/3.7/library/multiprocessing.html>`_ module.
    
    The can greatly improve training times as multiple CPU cores can process batch images in the background while
    training is done in the foreground on the GPU(s). 
    (The standard Keras ImageDataGenerator module processes batch images then trains serially)
    
    From the outside, this module works the exact same as `Keras ImageDataGenerator <https://keras.io/preprocessing/image>`_.

    Args:
        cores: The number of CPU cores to use for spawned image batch processes.
            This number can be either an integer, which specifies the exact number
            of CPU cores, or it can be a float < 1.0. The float is the percentage
            of CPU cores to use for processing.
            A large number of CPU cores will consume more system memory.

        debug: Use a ThreadPool rather than a Multiprocessing Pool,
            this allows for single-step debugging the processing function
                   
        max_batches_pending: This is the number of processed batches to queue.
            A larger number can improving training times at the expense of 
            increased system memory usage.
                  
        validation_augmentation_enabled: If True, then augmentations will be applied to
            validation data. If False, then no augmentations will be applied to validation data.

        featurewise_center: Boolean.
            Set input mean to 0 over the dataset, feature-wise.

        samplewise_center: Boolean. Set each sample mean to 0.

        featurewise_std_normalization: Boolean.
            Divide inputs by std of the dataset, feature-wise.

        samplewise_std_normalization: Boolean. Divide each input by its std.

        zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.

        zca_whitening: Boolean. Apply ZCA whitening.

        rotation_range: Int. Degree range for random rotations.

        width_shift_range: Float, 1-D array-like or int

            * float: fraction of total width, if < 1, or pixels if >= 1.
            * 1-D array-like: random elements from the array.
            * int: integer number of pixels from interval
              ``(-width_shift_range, +width_shift_range)``
            * With ``width_shift_range=2`` possible values
              are integers ``[-1, 0, +1]``,
              same as with ``width_shift_range=[-1, 0, +1]``,
              while with ``width_shift_range=1.0`` possible values are floats
              in the interval [-1.0, +1.0). 

        height_shift_range: Float, 1-D array-like or int

            * float: fraction of total height, if < 1, or pixels if >= 1.
            * 1-D array-like: random elements from the array.
            * int: integer number of pixels from interval
              ``(-height_shift_range, +height_shift_range)``
            * With ``height_shift_range=2`` possible values
              are integers ``[-1, 0, +1]``,
              same as with ``height_shift_range=[-1, 0, +1]``,
              while with ``height_shift_range=1.0`` possible values are floats
              in the interval ``[-1.0, +1.0)``.

        brightness_range: Tuple or list of two floats. Range for picking
            a brightness shift value from.

        shear_range: Float. Shear Intensity
            (Shear angle in counter-clockwise direction in degrees)

        zoom_range: Float or [lower, upper]. Range for random zoom.
            If a float, ``[lower, upper] = [1-zoom_range, 1+zoom_range]``.

        channel_shift_range: Float. Range for random channel shifts.

        fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}.
            Default is 'nearest'.
            Points outside the boundaries of the input are filled
            according to thegiven mode:

            * **constant** ``kkkkkkkk|abcd|kkkkkkkk (cval=k)``
            * **nearest**  ``aaaaaaaa|abcd|dddddddd``
            * **reflect**  ``abcddcba|abcd|dcbaabcd``
            * **wrap**  ``abcdabcd|abcd|abcdabcd``

        cval: Float or Int.
            Value used for points outside the boundaries
            when ``fill_mode = "constant"``.

        horizontal_flip: Boolean. Randomly flip inputs horizontally.

        vertical_flip: Boolean. Randomly flip inputs vertically.

        rescale: rescaling factor. Defaults to None.
            If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided
            (after applying all other transformations).

        get_batch_function: function that should return the transformed batch.
            If this is omitted, then iterator.get_batches_of_transformed_samples() is used.
            NOTE: If this is supplied, then none of the other callbacks are used.
            This function should have the following signature:

            .. highlight:: python
            .. code-block:: python
               
               def get_batches_of_transformed_samples(
                  batch_index:int, 
                  filenames:List[str], 
                  classes:List[int], 
                  params:ParallelProcessParams
               ) -> Tuple[int, Tuple[np.ndarray, np.ndarray]]:
                   ...
                   return batch_index, (batch_x, batch_y)

        noaug_preprocessing_function: function that will be applied on each input.
            The function will run after the image is resized but before it is augmented or standardized.
            The function should take at least two arguments and return the processed image:

            .. highlight:: python
            .. code-block:: python

               def my_processing_func(
                   params: ParallelProcessParams, 
                   x : np.ndarray,
                   class_id: Optional[int],
                   filename: Optional[str],
                   batch_index: Optional[int],
                   batch_class_ids: Optional[List[int]],
                   batch_filenames: Optional[List[str]]
                ) -> np.ndarray:
                   ...
                   return processed_x

        preprocessing_function: function that will be applied on each input.
            The function will run after the image is resized (if ``interpolation != None``) and augmented but before it is standardized.
            The function should take at least two arguments and return the processed image:

            .. highlight:: python
            .. code-block:: python

               def my_processing_func(
                   params: ParallelProcessParams, 
                   x : np.ndarray,
                   class_id: Optional[int],
                   filename: Optional[str],
                   batch_index: Optional[int],
                   batch_class_ids: Optional[List[int]],
                   batch_filenames: Optional[List[str]]
                ) -> np.ndarray:
                  ...
                  return processed_x

        validation_split: Float. Fraction of images reserved for validation
            (strictly between 0 and 1).

        dtype: Dtype to use for the generated arrays.

        brightness_range: Tuple of two floats. Control the brightness of an image. 
            An enhancement factor of 0.0 gives a black image. 
            A factor of 1.0 gives the original image.

        contrast_range: Tuple of two floats. Control the contrast of an image, 
            similar to the contrast control on a TV set. 
            An enhancement factor of 0.0 gives a solid grey image. A factor of 1.0 gives the original image.
    
        noise: List one one or more of the following:

            * **gauss** - Gaussian-distributed additive noise.
            * **poisson** - Poisson-distributed noise generated from the data.
            * **s&p** - Replaces random pixels with 0 or 1.
            * **speckle** - Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.

            The noise type used will be randomly selected from the provided list per image
    
        random_transforms_enabled: Enable random data augmentations. Default True

        max_samples_per_class: The maximum number of samples to use for a given class. 
            If ``-1`` then use all available samples. Default -1.

        disable_gpu_in_subprocesses: Disable GPU usage in spawned subprocesses, default: true

        batch_size: Generated batch size. This overrides the value given to flow_from_directory()
             Set to ``-1`` to set the batch size to be the number of samples
        

    '''
    def __init__(
        self, *,
        cores=0.25,
        debug=False,
        max_batches_pending=4, 
        get_batch_function=None,
        preprocessing_function=None,
        noaug_preprocessing_function=None, 
        validation_augmentation_enabled=True,
        contrast_range=None,
        noise=None,
        random_transforms_enabled=True,
        max_samples_per_class = -1,
        disable_gpu_in_subprocesses=True,
        save_to_dir=None,
        save_prefix=None,
        save_format=None,
        batch_size=None,
        **kwargs
    ):
        ImageDataGenerator.__init__(self, **_remove_preprocessing_function_arg(kwargs))

        self.cores = cores
        self.debug = debug
        self.random_transforms_enabled = random_transforms_enabled
        self.validation_augmentation_enabled = validation_augmentation_enabled
        self.max_batches_pending = max_batches_pending
        self.get_batch_function = get_batch_function
        self.parallel_preprocessing_function = preprocessing_function
        self.parallel_noaug_preprocessing_function = noaug_preprocessing_function
        self.contrast_range = contrast_range
        self.noise = noise
        self.max_samples_per_class = max_samples_per_class
        self.disable_gpu_in_subprocesses = disable_gpu_in_subprocesses
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.batch_size = batch_size


    def flow_from_directory(
        self,
        directory,
        target_size=(256, 256),
        color_mode='rgb',
        classes=None,
        class_mode='categorical',
        batch_size=32,
        batch_shape=None,
        shuffle=True,
        shuffle_index_dir=None,
        seed=None,
        save_to_dir=None,
        save_prefix=None,
        save_format='png',
        follow_links=False,
        subset=None,
        interpolation='bilinear',
        list_valid_filenames_in_directory_function=None,
        class_counts:Dict[str,int]=None,
    ):
        """Create the ParallelImageDataGenerator with the given dataset directory
        
        Takes the path to a directory & generates batches of augmented data.

        Args:

            directory: string, path to the target directory. It should contain one subdirectory per class.
                Any PNG, JPG, BMP, PPM or TIF images inside each of the subdirectories directory tree will be included in the generator. 

            target_size: Tuple of integers (height, width), defaults to (256, 256). 
                The dimensions to which all images found will be resized.

            color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb". 
                Whether the images will be converted to have 1, 3, or 4 channels.

            classes: Required, list of class subdirectories (e.g. ['dogs', 'cats']).
  
            class_mode: One of "categorical", "binary", "sparse", "input", or None. 
                Default: "categorical". Determines the type of label arrays that are returned: 

                * **categorical** will be 2D one-hot encoded labels, 
                * **binary** will be 1D binary labels, "sparse" will be 1D integer labels, 
                * **input** will be images identical to input images (mainly used to work with autoencoders). 
                * **None** no labels are returned (the generator will only yield batches of image data, which is useful to use with model.predict()).

                Please note that in case of class_mode None, the data still needs to reside in a subdirectory of directory for it to work correctly.

            batch_size: Size of the batches of data (default: 32).

            batch_shape: Shape of the batches of data. If omitted, this defaults to (batch_size, target_size)

            shuffle: Whether to shuffle the data (default: True) If set to False, sorts the data in alphanumeric order.

            shuffle_index_dir: If given, the dataset directory will be shuffled the first time it is processed and
                and an index file containing the shuffled file names is generated at the directory specified 
                by ``shuffle_index_dir``. The index file is reused to maintain the shuffled order for subsequent processing.
                If ``None``, then the dataset samples are sorted alphabetically and saved to an index file in the dataset directory. 
                The alphabetical index file is used for subsequent processing.
                Default: ``None``

            seed: Optional random seed for shuffling and transformations.

            follow_links: Whether to follow symlinks inside class subdirectories (default: False).
        
            subset: Subset of data ("training" or "validation") if validation_split is set in ParallelAudioDataGenerator.

            interpolation: Interpolation method used to resample the image if the target size is different from that of the loaded image. 
                Supported methods are ``none``, ``nearest``, ``bilinear``, ``bicubic``, ``lanczos``, ``box``, and ``hamming``.
                If ``none`` is used, then the image is **not automatically resized**.
                By default, ``bilinear`` is used.
        
            list_valid_filenames_in_directory_function: This is a custom function called for each class,
                that should return a list of valid file names for the given class.
                It has the following function signature:

                .. highlight:: python
                .. code-block:: python

                    def list_valid_filenames_in_directory(
                        base_directory:str, 
                        search_class:str, 
                        white_list_formats:List[str], 
                        split:float, 
                        follow_links:bool, 
                        shuffle_index_directory:str
                    ) -> Tuple[str, List[str]]
                        ...
                        return search_class, filenames
                
            Returns:
                A DirectoryIterator yielding tuples of (x, y) where x is a numpy array containing a batch of images with
                shape (batch_size, target_size, channels) and y is a numpy array of corresponding labels.

        """

        save_to_dir = self.save_to_dir or save_to_dir
        save_prefix = self.save_prefix or save_prefix
        save_format = self.save_format or save_format

        if batch_shape is None:
            if color_mode == 'rgba':
                image_shape = tuple(target_size) + (4,)
            elif self.color_mode == 'rgb':
                image_shape = tuple(target_size) + (3,)
            else:
                image_shape = tuple(target_size) + (1,)

            batch_shape = (batch_size,) + image_shape
        else:
            batch_shape = tuple(batch_shape)

        # If the batch_size was specified in the constructor
        # then that should override everything else
        if self.batch_size  is not None:
            batch_shape = (self.batch_size,) + batch_shape[1:]

        if save_to_dir:
            os.makedirs(save_to_dir, exist_ok=True)

        return ParallelDirectoryIterator(
            directory,
            self,
            target_size=target_size,
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            dtype=self.dtype,
            data_format=self.data_format,
            batch_shape=batch_shape,
            shuffle=shuffle,
            shuffle_index_dir=shuffle_index_dir,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation,
            cores=self.cores,
            debug = self.debug,
            max_batches_pending=self.max_batches_pending,
            get_batch_function=self.get_batch_function,
            preprocessing_function=self.parallel_preprocessing_function,
            noaug_preprocessing_function=self.parallel_noaug_preprocessing_function,
            list_valid_filenames_in_directory_function=list_valid_filenames_in_directory_function,
            max_samples_per_class=self.max_samples_per_class,
            disable_gpu_in_subprocesses=self.disable_gpu_in_subprocesses,
            class_counts=class_counts
        )
        
    def flow(
        self,
        x, 
        y=None, 
        batch_size=32, 
        shuffle=True, 
        sample_weight=None, 
        seed=None,
        save_to_dir=None, 
        save_prefix='', 
        save_format='png',
        subset=None
    ):
        """Takes data & label arrays, generates batches of augmented data.

            Args:
                x: Input data. Numpy array of rank 4 or a tuple. If tuple, the first
                    element should contain the images and the second element another numpy
                    array or a list of numpy arrays that gets passed to the output without
                    any modifications. Can be used to feed the model miscellaneous data
                    along with the images. In case of grayscale data, the channels axis of
                    the image array should have value 1, in case of RGB data, it should
                    have value 3, and in case of RGBA data, it should have value 4.
                y: Labels.
                batch_size: Int (default: 32).
                shuffle: Boolean (default: True).
                sample_weight: Sample weights.
                seed: Int (default: None).
                save_to_dir: None or str (default: None). This allows you to optionally
                    specify a directory to which to save the augmented pictures being
                    generated (useful for visualizing what you are doing).
                save_prefix: Str (default: ``''``). Prefix to use for filenames of saved
                    pictures (only relevant if `save_to_dir` is set).
                save_format: one of "png", "jpeg", "bmp", "pdf", "ppm", "gif",
                    "tif", "jpg"
                    (only relevant if ``save_to_dir`` is set). Default: "png".
                subset: Subset of data (``"training"`` or ``"validation"``) if
                    ``validation_split`` is set in ``ImageDataGenerator``.

            Returns:

                An ``Iterator`` yielding tuples of ``(x, y)``
                    where ``x`` is a numpy array of image data
                    (in the case of a single image input) or a list
                    of numpy arrays (in the case with
                    additional inputs) and ``y`` is a numpy array
                    of corresponding labels. If 'sample_weight' is not None,
                    the yielded tuples are of the form ``(x, y, sample_weight)``.
                    If ``y`` is None, only the numpy array ``x`` is returned.
            
            Raises:
                ValueError: If the Value of the argument, ``subset`` is other than
                    "training" or "validation".
            """
        save_to_dir = self.save_to_dir or save_to_dir
        save_prefix = self.save_prefix or save_prefix
        save_format = self.save_format or save_format
        batch_size = self.batch_size  or batch_size

        if save_to_dir:
            os.makedirs(save_to_dir, exist_ok=True)

        return ImageDataGenerator.flow(
            self,
            x=x,
            y=y,
            batch_size=batch_size,
            shuffle=shuffle,
            sample_weight=sample_weight,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            subset=subset
        )
    
    
    @property
    def default_transform(self) -> dict:
        """Return default transformation parameters"""
        return copy.deepcopy({
            'theta': 0,
            'tx': 0,
            'ty': 0,
            'shear': 0,
            'zx' : 1,
            'zy' : 1,
            'flip_horizontal': False,
            'flip_vertical': False,
            'channel_shift_intensity': None,
            'brightness': None,
            'contrast' : None,
            'noise': None
        })
        
    def get_random_transform(self, img_shape, seed=None) -> dict:
        """Generate a random transformation"""
        if not self.random_transforms_enabled:
            return self.default_transform
        
        params = super().get_random_transform(img_shape)
        
        contrast = None
        if self.contrast_range:
            contrast = np.random.uniform(self.contrast_range[0], self.contrast_range[1])
    
        noise = None 
        if self.noise:
            noise = random.choice(self.noise)
    
    
        params['contrast'] = contrast
        params['noise'] = noise
        
        return params


    def apply_transform(self, x, transform_parameters):
        """Apply the given transformation to the given image"""
        x = super().apply_transform(x, transform_parameters)
        if transform_parameters.get('contrast') is not None:
            contrast = transform_parameters.get('contrast')
            x = array_to_img(x)
            x = imgenhancer_Contrast = ImageEnhance.Contrast(x)
            x = imgenhancer_Contrast.enhance(contrast)
            x = img_to_array(x)
        
        if transform_parameters.get('noise') is not None:
            x = self._apply_noise(x, transform_parameters.get('noise'))
        
        
        return x
        
    
    def _apply_noise(self, x, noise):
        if noise == "gauss":
            row, col, ch = x.shape
            mean = 0
            var = 0.1
            sigma = var**0.5
            gauss = np.random.normal(mean, sigma, (row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            x = x + gauss
        
        elif noise == "s&p":
            row, col, ch = x.shape
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(x)
            
            #if grayscale, ignore last dimension
            shape = x.shape
            if shape[-1] == 1: shape = shape[:-1]
            
            
            # Salt mode
            num_salt = np.ceil(amount * x.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in shape]
            out[tuple(coords)] = 255
            
            # Pepper mode
            num_pepper = np.ceil(amount* x.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in shape]
            out[tuple(coords)] = 0
            x = out
        
        elif noise == "poisson":
            vals = len(np.unique(x))
            vals = 2 ** np.ceil(np.log2(vals))
            x = np.random.poisson(x * vals) / float(vals)
 
        elif noise =="speckle":
            row, col, ch = x.shape
            gauss = np.random.randn(row,col,ch)
            gauss = gauss.reshape(row,col,ch)        
            x = x + x * gauss
        
        return x




def _remove_preprocessing_function_arg(kwargs):
    if 'preprocessing_function' in kwargs:
        del kwargs['preprocessing_function']
    return kwargs
