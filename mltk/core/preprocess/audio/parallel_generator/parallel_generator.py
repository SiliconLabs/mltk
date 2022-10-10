"""Parallel Audio Data Generator

Refer to `ParallelAudioDataGenerator` for more details
"""

import warnings
import os
import copy
from typing import Dict

import numpy as np

from mltk.utils.python import append_exception_msg
try:
    import librosa
except Exception as e:
    if os.name != 'nt' and 'sndfile library not found' in f'{e}':
        append_exception_msg(e, '\n\nTry running: sudo apt-get install libsndfile1\n')
    raise

from mltk.core.preprocess.audio.audio_feature_generator import (
    AudioFeatureGeneratorSettings, 
    AudioFeatureGenerator
)
from mltk.core.preprocess import utils as data_utils
from mltk.core.preprocess.utils import audio as audio_utils
from .directory_iterator import ParallelDirectoryIterator


class ParallelAudioDataGenerator:
    '''Parallel Audio Data Generator
    
    This class as a similar functionality to the `Keras ImageDataGenerator <https://keras.io/preprocessing/image>`_
    
    Except, instead of processing image files it processes audio files.
    
    Additionally, batch samples are asynchronously processed using the Python
    'multiprocessing' package. This allows for efficient use of multi-core systems
    as future batch samples can be concurrently processed while processed batches may be
    used for training.
     
    This class works as follows:
    
    1. Class instantiated with parameters (see below)
    2. :py:meth:`~flow_from_directory` called which lists each classes' samples in the specified directory
    3. The return value of :py:meth:`~flow_from_directory` is a 'generator' which should be given to a model fit() method
    4. During fitting, batches of samples are concurrently processed using the following sequence:
        a0. If get_batch_function() is given, then call this function an skip the rest of these steps
        a. Read sample raw audio file
        b. If supplied, call noaug_preprocessing_function()
        c. Generate random transform parameters based on parameters from step 1)  
        d. Trim silence from raw audio sample based on ``trim_threshold_db``
        e. Pad zeros before and after trimmed audio based on ``sample_length_seconds`` and ``offset_range``  
        f. Augment padded audio based on randomly generated transform parameters from part c)  
        g. If supplied, call preprocessing_function()
        h. If ``frontend_enabled=True``, pass augmented audio through :py:class:`mltk.core.preprocess.audio.audio_feature_generator.AudioFeatureGenerator` and return spectrogram as 2D array  
        i. If supplied, call postprocessing_function()
        j. If ``frontend_enabled=True``, normalize based on ``samplewise_center``, ``samplewise_std_normalization``, and ``rescale``
    
    Notes:

        * If ``_unknown_`` is added as a class to :py:meth:`~flow_from_directory`, then the generator will automatically
          add an 'unknown' class to the generated batches.
          Unused classes in the dataset directory will be randomly selected and used as an 'unknown' class.
          The other augmentation parameters will be applied to the 'unknown' samples. 
          Use the ``unknown_class_percentage`` setting to control the size of this class.
    
        * If ``_silence_`` is added as a class to :py:meth:`~flow_from_directory`, then the generator will automatically
          add 'silence' samples is all zeros with the background noise augmentations added.
          Use the ``silence_class_percentage`` setting to control the size of this class.
    
    Args:

        cores: The number of CPU cores to use for spawned audio processing batch processes.
            This number can be either an integer, which specifies the exact number
            of CPU cores, or it can be a float < 1.0. The float is the percentage
            of CPU cores to use for processing.
            A large number of CPU cores will consume more system memory.

        debug: If true then use the Python threading library instead of multiprocessing
            This is useful for debugging as it allows for single-stepping in the generator threads
            and callback functions
                   
        max_batches_pending:  This is the number of processed batches to queue.
            A larger number can improving training times at the expense of 
            increased system memory usage.

        get_batch_function: function that should return the transformed batch.
            If this is omitted, then iterator.get_batches_of_transformed_samples() is used
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
            The function will run before any augmentation is done on the audio sample.
            The 'x' argument is of the shape [sample_length] and is a float32 scaled between (-1,1).
            See https://librosa.org/doc/main/generated/librosa.load.html
            The function should take at least two arguments:

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
            The function will run after the audio is augmented but before it does through the AudioFeatureGenerator (if enabled).
            The 'x' argument is of the shape [sample_length] and is a float32 scaled between [-1,1].
            See `librosa.load() <https://librosa.org/doc/main/generated/librosa.load.html>`_
            The function should take at least two arguments, and return the processed sample:.

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

        postprocessing_function: function that will be applied on each input.
            The function will run after the audio is passed through the AudioFeatureGenerator (if enabled).
            So the 'x' argument is a spectrogram of the shape: [height, width, 1].
            The function should take at least two arguments and return the processed sample [height, width, 1]:

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
                 
        samplewise_center: Center each sample's processed data about its norm

        samplewise_std_normalization: Divide processed sample data by its STD

        samplewise_normalize_range: Normalize the input by the values in this range
            For instance, if samplewise_normalize_range=(0,1), then each input will be scaled to values between 0 and 1
            Note that this normalization is applied after all other enabled normalizations

        rescale: Divide processed sample data by value

        validation_split: Percentage of sample data to use for validation

        validation_augmentation_enabled: If True, then augmentations will be applied to
            validation data. If False, then no augmentations will be applied to validation data.

        dtype: Output data type for the x samples
            Default: float32  

        frontend_dtype: Output data format of the audio frontend. If omitted then default to the ``dtype`` argument. 
            This is only used if ``frontend_enabled=True``. 

            * **uint16:*** This the raw value generated by the internal AudioFeatureGenerator library
            * **float32:** This is the uint16 value directly casted to a float32
            * **int8:*** This is the int8 value generated by the TFLM "micro features" library.

            Refer to :py:class:`mltk.core.preprocess.audio.audio_feature_generator.AudioFeatureGenerator` for more details.

        trim_threshold_db: Use to trim silence from samples; the threshold (in decibels) below reference to consider as silence

        noise_colors: List of noise colors to randomly add to samples, possible options: 

            * ['white', 'brown', 'blue', 'pink', 'violet']
            * OR 'all' to use all
            * OR 'none' to use none
            
        noise_color_range: Tuple (min, max) for randomly selecting noise color's loudness, 0.0 no noise, 1.0 100% noise

        speed_range:  Tuple (min, max) for randomly augmenting audio's speed, < 1.0 slow down, > 1.0 speed up

        pitch_range: Tuple (min, max) for randomly augmenting audio's pitch, < 0 lower pitch, > 0 higher pitch
            This can either be an integer or float. An integer represents the number of semitone steps.
            A float is converted to semitone steps <float>*12, so for example, a range of (-.5,.5) is converted to (-6,6)

        vtlp_range: Tuple (min, max) for randomly augmenting audio's vocal tract length perturbation

        loudness_range: Tuple (min, max) for randomly augmenting audio's volume, < 1.0 decrease loudness, > 1.0 increase loudness

        bg_noise_range: Tuple (min, max) for randomly selecting background noise's loudness, < 1.0 decrease loudness, > 1.0 increase loudness

        bg_noise_dir: Path to directory containing background noise audio files. A bg noise file will be randomly selected and cropped then applied to samples.
            .. note:: If noise_colors is also supplied then either a bg_noise or noise_color will randomly be applied to each sample

        offset_range: Tuple (min, max) for randomly selecting the offset of where to pad a sample to @ref sample_length_seconds.
            For instance, if offset_range=(0.0, 1.0), then  

            * trimmed_audio     = trim(raw_audio, trim_threshold_db) # Trim silence  
            * required_padding  = (sample_length_seconds * sample_rate) - len(trimmed_audio)  
            * pad_upto_index    = required_padding * random.uniform(offset_range[0], offset_range[1])  
            * padded_audio      = concat(zeros * pad_upto_index, trimmed_audio, zeros * (required_padding - pad_upto_index))

        unknown_class_percentage: If an ``_unknown_`` class is added to the class list, then 'unknown' class samples will automatically
            be added to batches. This specifies the percentage of of samples to generate relative the smallest number
            of samples of other classes. For instance, if another class has 1000 samples and unknown_class_percentage=0.8,
            then the number of 'unknown' class samples generated will be 800.

        silence_class_percentage: If a ``_silence_`` class is added to the class list, then 'silence' class samples will automatically
            be added to batches. This specifies the percentage of of samples to generate relative the smallest number
            of samples of other classes. For instance, if another class has 1000 samples and silence_class_percentage=0.8,
            then the number of 'silence' class samples generated will be 800.

        disable_random_transforms: Disable random data augmentations 

        frontend: AudioFeatureGenerator settings, see :py:class:`mltk.core.preprocess.audio.audio_feature_generator.AudioFeatureGeneratorettings` for more details

        frontend_enabled: By default, the frontend is enabled. After augmenting audio sample, pass it through the AudioFeatureGenerator and 
            return the generated spectrogram.
            If disabled, after augmenting audio sample, return the 1D sample. In this case it is recommended to use the ``postprocessing_function``
            callback to convert the samples to the required shape and data type.
            NOTE: You must also specify the ``sample_shape`` parameter if ``frontend_enabled=False``

        sample_shape: The shape of the generated sample. This is only used/required if ``frontend_enabled=False``

        disable_gpu_in_subprocesses: Disable GPU usage in spawned subprocesses, default: true

        add_channel_dimension: If true and ``frontend_enabled=True``, then automatically convert 
            generated sample shape from [height, width] to [height, width, 1]. 
            If false, then generated sample shape is [height, width].
    
    '''
    def __init__(
        self, 
        cores=0.25,
        debug=False,
        max_batches_pending=4, 
        get_batch_function=None,
        noaug_preprocessing_function=None, 
        preprocessing_function=None,
        postprocessing_function=None,
        samplewise_center=False,
        samplewise_std_normalization=False,
        samplewise_normalize_range=None,
        rescale=None,
        validation_split=0.0,
        validation_augmentation_enabled=True,
        dtype='float32',
        frontend_dtype=None,
        trim_threshold_db=20, 
        noise_colors=None,
        noise_color_range=None,
        speed_range=None,
        pitch_range=None,
        vtlp_range=None,
        loudness_range=None,
        bg_noise_range=None,
        bg_noise_dir=None,
        offset_range=(0.0,1.0),
        unknown_class_percentage=1.0,
        silence_class_percentage=0.6,
        disable_random_transforms=False,
        frontend_settings: AudioFeatureGeneratorSettings = None,
        frontend_enabled = True,
        sample_shape=None,
        disable_gpu_in_subprocesses=True,
        add_channel_dimension=True
    ):

        self.cores = cores
        self.debug = debug
        self.disable_random_transforms = disable_random_transforms
        self.max_batches_pending = max_batches_pending
        self.get_batch_function = get_batch_function
        self.noaug_preprocessing_function = noaug_preprocessing_function
        self.preprocessing_function = preprocessing_function
        self.postprocessing_function = postprocessing_function
        self.samplewise_center = samplewise_center
        self.samplewise_std_normalization = samplewise_std_normalization
        self.samplewise_normalize_range = samplewise_normalize_range
        self.rescale = rescale
        self.validation_split = validation_split
        self.validation_augmentation_enabled = validation_augmentation_enabled
        self.dtype = dtype
        self.frontend_dtype = frontend_dtype
        self.trim_threshold_db = trim_threshold_db
        self.speed_range = speed_range
        self.pitch_range = pitch_range
        self.loudness_range = loudness_range
        self.vtlp_range = vtlp_range
        self.offset_range = offset_range
        self.unknown_class_percentage = unknown_class_percentage
        self.silence_class_percentage = silence_class_percentage

        if frontend_settings is None:
            raise Exception('Missing "frontend_settings" parameter. You must specify the AudioFeatureGenerator settings')

        self.frontend_settings = frontend_settings
        self.frontend = None
        self.frontend_enabled = frontend_enabled
        if not frontend_enabled and sample_shape is None:
            raise RuntimeError('Must provide "sample_shape" parameter if frontend_enabled=False')
        self._sample_shape = sample_shape
        self.disable_gpu_in_subprocesses = disable_gpu_in_subprocesses
        self.add_channel_dimension = add_channel_dimension

        
        self.NOISE_COLORS =  ('white', 'brown', 'blue', 'pink', 'violet')
        if noise_colors == 'all':
            noise_colors = self.NOISE_COLORS
        self.noise_colors = noise_colors
        self.noise_color_range = noise_color_range
        self.bg_noise_range = bg_noise_range
        self.bg_noises = None
        self.bg_noise_dir = bg_noise_dir

        if samplewise_std_normalization:
            if not samplewise_center:
                self.samplewise_center = True
                warnings.warn('This ParallelAudioDataGenerator specifies '
                              '`samplewise_std_normalization`, '
                              'which overrides setting of '
                              '`samplewise_center`.')


    @property 
    def sample_shape(self) -> tuple:
        """The shape of the sample as a tuple"""
        if self.frontend_enabled:
            return self.frontend_settings.spectrogram_shape
        return self._sample_shape
    @sample_shape.setter
    def sample_shape(self, v):
        raise RuntimeError('The sample_shape is calculated dynamically, it cannot be manually set')


    @property
    def sample_length(self) -> int:
        """Return the length of the audio sample as the number of individual ADC samples"""
        return int((self.sample_length_ms * self.sample_rate_hz) / 1000)
    @sample_length.setter
    def sample_length(self, v):
        raise RuntimeError('The sample_length is calculated dynamically, it cannot be manually set. Try setting sample_length_ms')

    @property
    def sample_length_ms(self) -> int:
        """Return the AudioFeatureGeneratorSettings.sample_length_ms value"""
        return self.frontend_settings.sample_length_ms
    @sample_length_ms.setter 
    def sample_length_ms(self, v: int):
        self.frontend_settings.sample_length_ms = v

    @property
    def sample_rate_hz(self) -> int:
        """Return the AudioFeatureGeneratorSettings.sample_rate_hz value"""
        return self.frontend_settings.sample_rate_hz
    @sample_rate_hz.setter 
    def sample_rate_hz(self, v: int):
        self.frontend_settings.sample_rate_hz = v


    def flow_from_directory(
        self,
        directory,
        classes,
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        shuffle_index_dir=None,
        seed=None,
        follow_links=False,
        subset=None,
        max_samples_per_class=-1,
        list_valid_filenames_in_directory_function=None,
        class_counts:Dict[str,int]=None,
        **kwargs
    ):
        """Create the ParallelAudioGenerator with the given dataset directory
        
        Takes the path to a directory & generates batches of augmented data.

        Args:

            directory: string, path to the target directory. It should contain one subdirectory per class. 
                Any PNG, JPG, BMP, PPM or TIF images inside each of the subdirectories directory tree will be included in the generator. 

            classes: Required, list of class subdirectories (e.g. ['dogs', 'cats'])

                * If ``_unknown_`` is added as a class then the generator will automatically add an 'unknown' class to the generated batches. 
                  Unused classes in the dataset directory will be randomly selected and used as an 'unknown' class.
                  The other augmentation parameters will be applied to the 'unknown' samples.
                * If ``_silence_`` is added as a class then the generator will automatically add 'silence' samples with all zeros 
                  with the background noise augmentations added.
  
            class_mode: One of "categorical", "binary", "sparse", "input", or None. 
                Default: "categorical". Determines the type of label arrays that are returned: 

                * **categorical** will be 2D one-hot encoded labels, 
                * **binary** will be 1D binary labels, "sparse" will be 1D integer labels, 
                * **input** will be images identical to input images (mainly used to work with autoencoders). 
                * **None** no labels are returned (the generator will only yield batches of image data, which is useful to use with model.predict()).  
                
                Please note that in case of class_mode None, the data still needs to reside in a subdirectory of directory for it to work correctly.

            batch_size: Size of the batches of data (default: 32).

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

            max_samples_per_class: The maximum number of samples to use for a given class. If ``-1`` then use all available samples.
        
            list_valid_filenames_in_directory_function: This is a custom function called for each class,
                that should return a list of valid file names for the given class.
                It has the following function signature:

                .. highlight:: python
                .. code-block:: python
                    
                    def list_valid_filenames_in_directory(
                            base_directory:str, 
                            search_class:str, 
                            white_list_formats:List[str], 
                            split:Tuple[float,float], 
                            follow_links:bool, 
                            shuffle_index_directory:str
                    ) -> Tuple[str, List[str]]
                        ...
                        return search_class, filenames
                    

        Returns:

            A DirectoryIterator yielding tuples of (x, y) where x is a numpy array containing a batch of images with 
            shape (batch_size, target_size, channels) and y is a numpy array of corresponding labels.

        """
        
        if self.bg_noise_dir:
            bg_noise_dir = self.bg_noise_dir
            if not os.path.exists(bg_noise_dir):
                bg_noise_dir = f'{directory}/{self.bg_noise_dir}'
            if not os.path.exists(bg_noise_dir):
                raise Exception(f'bg_noise_dir not found {self.bg_noise_dir}')
            
            self.bg_noises = {}
            for fn in os.listdir(bg_noise_dir):
                if not fn.endswith('.wav'):
                    continue 
                fp = os.path.join(bg_noise_dir, fn)
                noise_audio, _ = librosa.load(fp, sr=self.sample_rate_hz, mono=True, dtype='float32')
                self.bg_noises[fn[:-4]] = noise_audio

        return ParallelDirectoryIterator(
            directory,
            self,
            classes=classes,
            unknown_class_percentage=self.unknown_class_percentage,
            silence_class_percentage=self.silence_class_percentage,
            sample_shape=self.sample_shape,
            class_mode=class_mode,
            batch_size=batch_size,
            sample_rate=self.sample_rate_hz,
            sample_length_ms=self.sample_length_ms,
            shuffle=shuffle,
            shuffle_index_dir=shuffle_index_dir,
            seed=seed,
            dtype=self.dtype,
            frontend_dtype=self.frontend_dtype,
            follow_links=follow_links,
            subset=subset,
            cores=self.cores,
            debug=self.debug,
            max_batches_pending=self.max_batches_pending,
            get_batch_function=self.get_batch_function,
            noaug_preprocessing_function=self.noaug_preprocessing_function,
            preprocessing_function=self.preprocessing_function,
            postprocessing_function=self.postprocessing_function,
            list_valid_filenames_in_directory_function=list_valid_filenames_in_directory_function,
            max_samples_per_class=max_samples_per_class,
            frontend_enabled=self.frontend_enabled,
            disable_gpu_in_subprocesses=self.disable_gpu_in_subprocesses,
            add_channel_dimension=self.add_channel_dimension,
            class_counts=class_counts
        )
    
    
    @property
    def default_transform(self) -> dict:
        """Retrun the default augmentations transform settings"""
        return copy.deepcopy({
            'offset_percentage': 0.0,
            'loudness_factor': 1.0,
            'noise_color': 'none',
            'noise_color_factor': 0.0,
            'bg_noise': None,
            'bg_noise_factor': 0.0,
            'speed_factor': 1.0,
            'pitch_factor': 0,
            'vtlp_factor': 1.0
        })
    
    
    def get_random_transform(self) -> dict:
        """Generate random augmentation settings based on the configeration parameters"""
        xform_params = self.default_transform
        if self.disable_random_transforms:
            return xform_params
        
        if self.offset_range:
            xform_params['offset_percentage'] = np.random.uniform(self.offset_range[0], self.offset_range[1])
        
        else:
            xform_params['offset_percentage'] = np.random.uniform(0.0, 1.0)
        
        
        if self.loudness_range:
            xform_params['loudness_factor'] =  np.random.uniform(self.loudness_range[0], self.loudness_range[1])
    
        if self.noise_colors:
            xform_params['noise_color'] = np.random.choice(self.noise_colors) 
            if self.noise_color_range:
                xform_params['noise_color_factor'] = np.random.uniform(self.noise_color_range[0], self.noise_color_range[1])
            else:
                xform_params['noise_color_factor'] = 0.0
    
        if self.bg_noises:
            if self.bg_noise_range:
                xform_params['bg_noise_factor'] = np.random.uniform(self.bg_noise_range[0], self.bg_noise_range[1])
            else:
                xform_params['bg_noise_factor'] = 0.0
            
            xform_params['bg_noise'] = np.random.choice([*self.bg_noises])
        
        if not xform_params['noise_color'] is None and not xform_params['bg_noise'] is None:
            use_noise_color = np.random.randint(0, 1)
            if use_noise_color < 0.5:
                xform_params['bg_noise'] = None 
            else:
                xform_params['noise_color'] = None 
                
    
        if self.speed_range:
            speeds = [round(i, 1) for i in np.arange(self.speed_range[0], self.speed_range[1], 0.1)]
            speeds = [s for s in speeds if s != 1.0]
            xform_params['speed_factor'] = speeds[np.random.randint(len(speeds))]
    
        if self.pitch_range:
            def _convert_range(v):
                if not isinstance(v, int):
                    return int(v*12)
                return v
            lower = _convert_range(self.pitch_range[0])
            upper = _convert_range(self.pitch_range[1])
            xform_params['pitch_factor'] = np.random.randint(lower, upper)
    
        if self.vtlp_range:
            xform_params['vtlp_factor'] = np.random.randint(self.vtlp_range[0], self.vtlp_range[1])
            
            
        return xform_params 
    
    
    def adjust_length(self, sample, orignal_sr, offset=0.0, whole_sample=False, out_length=None):
        """Adjust the audio sample length to fit the sample_length_seconds parameter
        This will pad with zeros or crop the input sample as necessary
        """

        if whole_sample:
            if orignal_sr != self.sample_rate_hz:
                sample = librosa.core.resample(sample, orig_sr=orignal_sr, target_sr=self.sample_rate_hz)
            return sample

        out_length = out_length or self.sample_length

        return audio_utils.adjust_length(
            sample=sample, 
            target_sr=self.sample_rate_hz,
            original_sr=orignal_sr,
            out_length=out_length,
            offset=offset,
            trim_threshold_db=self.trim_threshold_db
        )
    

    def apply_transform(self, sample, orignal_sr, params, whole_sample=False):
        """Apply the given transform parameters to the input audio sample"""
        if params['speed_factor'] and params['speed_factor'] != 1.0:
            rate =  params['speed_factor']
            sample = librosa.effects.time_stretch(sample, rate=rate)
            
        if params['pitch_factor'] and params['pitch_factor'] != 0:
            sample = librosa.effects.pitch_shift(sample, sr=orignal_sr, n_steps=params['pitch_factor'])
        
        if params['vtlp_factor'] and params['vtlp_factor'] != 1.0:
            sample = self._apply_vtlp(sample, orignal_sr, params['vtlp_factor'])
        
        sample = self.adjust_length(sample, orignal_sr, offset=params['offset_percentage'], whole_sample=whole_sample)

        if params['loudness_factor'] and params['loudness_factor'] != 1.0:
            sample = sample * params['loudness_factor']


        if params['noise_color'] and params['noise_color_factor'] > 0:
            color = params['noise_color'].lower()
            if color != 'none':
                noise = self._get_color_noise(sample, color)
                sample = sample + (noise * (params['noise_color_factor'] * params['loudness_factor']))
        
        if not params['bg_noise'] is None and params['bg_noise_factor'] > 0:
            bg_noise_offset = np.random.uniform(0.0, 1.0)
            bg_noise = self.bg_noises[params['bg_noise']]
            bg_noise = self.adjust_length(bg_noise, orignal_sr, offset=bg_noise_offset, out_length=len(sample))
            # bg_noise = np.clip(bg_noise, sample.min(), sample.max())

            sample = sample + (bg_noise * (params['bg_noise_factor'] * params['loudness_factor']))

        return sample


    def apply_frontend(self, sample, dtype=np.float32) -> np.ndarray:
        """Send the audio sample through the AudioFeatureGenerator and return the generated spectrogram"""
        return audio_utils.apply_frontend(
            sample=sample,
            settings=self.frontend_settings,
            dtype=dtype
        )

    
    def standardize(self, sample):
        """Applies the normalization configuration in-place to a batch of inputs.

        Args:
            sample: Input sample to normalize
            rescale: ``sample *= rescale``
            samplewise_center: ``sample -= np.mean(sample, keepdims=True)``
            samplewise_std_normalization: ``sample /= (np.std(sample, keepdims=True) + 1e-6)``
            samplewise_normalize_range: ``sample = diff * (sample - np.min(sample)) / np.ptp(sample) + lower``
            dtype: The output dtype, if not dtype if given then sample is converted to float32
        Returns:
            The normalized value of sample
        """
        return data_utils.normalize(
            x=sample, 
            rescale=self.rescale,
            samplewise_center=self.samplewise_center,
            samplewise_std_normalization=self.samplewise_std_normalization,
            samplewise_normalize_range=self.samplewise_normalize_range,
            dtype=self.dtype
        )



        
    def _apply_vtlp(self, sample, original_sr, factor):
        stft = librosa.core.stft(sample)
        time_dim, freq_dim = stft.shape
        data_type = type(stft[0][0])

        factors = self._get_vtlp_scale_factors(freq_dim, original_sr, alpha=factor)
        factors *= (freq_dim - 1) / max(factors)
        new_stft = np.zeros([time_dim, freq_dim], dtype=data_type)

        for i in range(freq_dim):
            # first and last freq
            if i == 0 or i + 1 >= freq_dim:
                new_stft[:, i] += stft[:, i]
            else:
                warp_up = factors[i] - np.floor(factors[i])
                warp_down = 1 - warp_up
                pos = int(np.floor(factors[i]))

                new_stft[:, pos] += warp_down * stft[:, i]
                new_stft[:, pos+1] += warp_up * stft[:, i]

        return librosa.core.istft(new_stft)
        
        
    def _get_vtlp_scale_factors(self, freq_dim, sampling_rate, fhi=4800, alpha=0.9):
        factors = []
        freqs = np.linspace(0, 1, freq_dim)
        
        fhi = min(fhi, (sampling_rate - 1) // 2)

        scale = fhi * min(alpha, 1)
        f_boundary = scale / alpha
        half_sr = sampling_rate / 2

        for f in freqs:
            f *= sampling_rate
            if f <= f_boundary:
                factors.append(f * alpha)
            else:
                warp_freq = half_sr - (half_sr - scale) / (half_sr - scale / alpha) * (half_sr - f)
                factors.append(warp_freq)

        return np.array(factors)


    def _get_color_noise(self, sample, color):
        # https://en.wikipedia.org/wiki/Colors_of_noise
        sample_length = len(sample)
        uneven = sample_length % 2
        fft_size = sample_length // 2 + 1 + uneven
        noise_fft = np.random.randn(fft_size)
        color_noise = np.linspace(1, fft_size, fft_size)
        scale_factor = 1.0

        if color == 'white':
            pass  # no color noise
        
        elif color == 'pink':
            color_noise = color_noise ** (-1)  # 1/f
            scale_factor = 10
            
        elif color in 'brown':
            color_noise = color_noise ** (-2)  # 1/f^2
            scale_factor = 10
            
        elif color in 'blue':
            scale_factor = 0.01
            
        elif color in 'violet':
            color_noise = color_noise ** 2  # f^2
            scale_factor = 0.01

        else:
            raise Exception('Unknown noise color: {}, valid colors: {}'.format(color, ','.join(self.NOISE_COLORS)))

        if color != 'white':
            noise_fft = noise_fft * color_noise

        if uneven:
            noise_fft = noise_fft[:-1]

        noise = np.fft.irfft(noise_fft)
        
        noise = np.clip(noise, sample.min(), sample.max()) * scale_factor
        if len(noise) != sample_length:
            old_noise = noise 
            noise = np.zeros((sample_length,), dtype=np.float32)
            noise[:len(old_noise)] = old_noise


        return noise.astype(sample.dtype)
    
