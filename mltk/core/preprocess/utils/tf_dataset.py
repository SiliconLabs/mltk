"""Utilities for processing `Tensorflow Datasets <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>`_"""
import os
from typing import Callable, Tuple, List, Dict, Union
import wave
import tensorflow as tf
import numpy as np

from mltk.core import get_mltk_logger
from mltk.utils.path import create_tempdir
from mltk.utils.process_pool import (ProcessPool, calculate_n_jobs)


from .list_directory import list_dataset_directory
from .audio import read_audio_file
from .image import read_image_file


def load_audio_directory(
    directory:str,
    classes:List[str],
    unknown_class_percentage:float=1.0,
    silence_class_percentage:float=1.0,
    class_counts:Dict[str,int] = None,
    onehot_encode=False,
    shuffle:bool=False,
    seed=None,
    split:Tuple[float,float]=None,
    max_samples_per_class:int=-1,
    sample_rate_hz:int=None,
    return_audio_data=False,
    return_audio_sample_rate=False,
    white_list_formats:List[str]=None,
    follow_links=False,
    shuffle_index_directory:str=None,
    list_valid_filenames_in_directory_function:Callable=None,
    process_samples_function:Callable[[str,Dict[str,str]],None]=None
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load a directory of audio samples and return a tuple of `Tensorflow Datasets <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>`_ (samples, label_ids)

    The given audio directory should have the structure::

        <class1>/sample1.wav
        <class1>/sample2.wav
        ...
        <class1>/optional sub directory/sample9.wav
        <class2>/sample1.wav
        <class2>/sample2.wav
        ...
        <class3>/sample1.wav
        <class3>/sample2.wav

    Where each <class> is found in the given ``classes`` argument.

    .. seealso::
        See the `Tensor Dataset API <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>`_
        for more details of how to use the returned datasets

    Args:
        directory: Directory path to audio dataset
        classes: List of class labels to include in the returned dataset

            * If ``_unknown_`` is added as an entry to the ``classes``, then this API will automatically
              add an 'unknown' class to the generated batches.
              Unused classes in the dataset directory will be randomly selected and used as an 'unknown' class.
              Use the ``unknown_class_percentage`` setting to control the size of this class.
            * If ``_silence_`` is added as an entry to the ``classes``, then this API will automatically
              add 'silence' samples with all zeros.
              Use the ``silence_class_percentage`` setting to control the size of this class.

        unknown_class_percentage: If an ``_unknown_`` class is added to the class list, then 'unknown' class samples will automatically
            be added to batches. This specifies the percentage of of samples to generate relative the smallest number
            of samples of other classes. For instance, if another class has 1000 samples and unknown_class_percentage=0.8,
            then the number of 'unknown' class samples generated will be 800.
            Set this parameter to None to disable this feature
        silence_class_percentage: If a ``_silence_`` class is added to the class list, then 'silence' class samples will automatically
            be added to batches. This specifies the percentage of of samples to generate relative the smallest number
            of samples of other classes. For instance, if another class has 1000 samples and silence_class_percentage=0.8,
            then the number of 'silence' class samples generated will be 800.
            Set this parameter to None to disable this feature
        class_counts: Dictionary which will be populated with the sample counts for each class
        onehot_encode: If true then the audio labels are onehot-encoded
            If false, then only the class id (corresponding to it index in the ``classes`` argument) is returned
        shuffle: If true, then shuffle the dataset
        seed: The seed to use for shuffling the dataset
        split: A tuple indicating the (start,stop) percentage of the dataset to return,
            e.g. (.75, 1.0) -> return last 25% of dataset
            If omitted then return the entire dataset
        max_samples_per_class: Maximum number of samples to return per class, this can be useful for debug to limit the dataset size
        sample_rate_hz: Sample rate to convert audio samples, if omitted then return native sample rate
        return_audio_data: If true then the audio file data is returned, if false then only the audio file path is returned
        return_audio_sample_rate: If true and return_audio_data is True, then the audio file data and corresponding sample rate is returned, if false then only the audio file data is return
        white_list_formats: List of file extensions to include in the search.
            If omitted then only return ``.wav`` files
        follow_links: If true then follow symbolic links when recursively searching the given dataset directory
        shuffle_index_directory: Path to directory to hold generated index of the dataset
            If omitted, then an index is generated at <directory>/.index
        list_valid_filenames_in_directory_function: This is a custom dataset processing callback.
            It should return a list of valid file names for the given class.
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

        process_samples_function: This allows for processing the samples BEFORE they're returned by this API.
            This allows for adding/removing samples.
            It has the following function signature:

            .. highlight:: python
            .. code-block:: python

                def process_samples(
                    directory:str, # The provided directory to this API
                    sample_paths:Dict[str,str], # A dictionary: <class name>, [<sample paths relative to directory>]
                    split:Tuple[float,float],
                    follow_links:bool,
                    white_list_formats:List[str],
                    shuffle:bool,
                    seed:int,
                    **kwargs
                )
                    ...

    Returns:
        Returns a tuple of two tf.data.Dataset, (samples, label_ids)
    """

    white_list_formats = white_list_formats or ('.wav',)

    sample_paths, sample_labels = list_dataset_directory(
        directory=directory,
        classes=classes,
        max_samples_per_class=max_samples_per_class,
        list_valid_filenames_in_directory_function=list_valid_filenames_in_directory_function,
        shuffle_index_directory=shuffle_index_directory,
        unknown_class_percentage=unknown_class_percentage,
        unknown_class_label='_unknown_',
        empty_class_percentage=silence_class_percentage,
        empty_class_label = '_silence_',
        split=split,
        shuffle=shuffle,
        seed=seed,
        white_list_formats=white_list_formats,
        return_absolute_paths=True,
        follow_links=follow_links,
        class_counts=class_counts,
        process_samples_function=process_samples_function
    )

    _update_silence_samples(sample_paths, sample_rate_hz)

    label_ds = tf.data.Dataset.from_tensor_slices(np.array(sample_labels, dtype=np.int32))
    path_ds = tf.data.Dataset.from_tensor_slices(sample_paths)

    if return_audio_data:
        feature_ds = path_ds.map(
            lambda x: read_audio_file(x, return_numpy=False, return_sample_rate=return_audio_sample_rate)
        )

    else:
        feature_ds = path_ds

    if onehot_encode:
        label_ds = label_ds.map(
            lambda x: tf.one_hot(x, depth=len(classes), dtype=tf.int32),
        )

    return feature_ds, label_ds


def load_image_directory(
    directory:str,
    classes:List[str],
    unknown_class_percentage:float=1.0,
    class_counts:Dict[str,int] = None,
    onehot_encode=False,
    shuffle:bool=False,
    seed=None,
    split:Tuple[float,float]=None,
    max_samples_per_class:int=-1,
    return_image_data = False,
    white_list_formats:List[str]=None,
    follow_links=False,
    shuffle_index_directory:str=None,
    list_valid_filenames_in_directory_function=None,
    process_samples_function=None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load a directory of images samples and return a tuple of `Tensorflow Datasets <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>`_ (samples, label_ids)

    The given images directory should have the structure::

        <class1>/sample1.png
        <class1>/sample2.jpg
        ...
        <class1>/optional sub directory/sample9.png
        <class2>/sample1.jpg
        <class2>/sample2.jpg
        ...
        <class3>/sample1.jpg
        <class3>/sample2.jpg

    Where each <class> is found in the given ``classes`` argument.

    .. seealso::
        See the `Tensor Dataset API <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>`_
        for more details of how to use the returned datasets

    Args:
        directory: Directory path to images dataset
        classes: List of class labels to include in the returned dataset

            * If ``_unknown_`` is added as an entry to the ``classes``, then this API will automatically
              add an 'unknown' class to the generated batches.
              Unused classes in the dataset directory will be randomly selected and used as an 'unknown' class.
              Use the ``unknown_class_percentage`` setting to control the size of this class.

        unknown_class_percentage: If an ``_unknown_`` class is added to the class list, then 'unknown' class samples will automatically
            be added to batches. This specifies the percentage of of samples to generate relative the smallest number
            of samples of other classes. For instance, if another class has 1000 samples and unknown_class_percentage=0.8,
            then the number of 'unknown' class samples generated will be 800.
            Set this parameter to None to disable this feature
        class_counts: Dictionary which will be populated with the sample counts for each class
        onehot_encode: If true then the audio labels are onehot-encoded
            If false, then only the class id (corresponding to it index in the ``classes`` argument) is returned
        shuffle: If true, then shuffle the dataset
        seed: The seed to use for shuffling the dataset
        split: A tuple indicating the (start,stop) percentage of the dataset to return,
            e.g. (.75, 1.0) -> return last 25% of dataset
            If omitted then return the entire dataset
        max_samples_per_class: Maximum number of samples to return per class, this can be useful for debug to limit the dataset size
        return_image_data: If true then the image file data is returned, if false then only the image file path is returned
        white_list_formats: List of file extensions to include in the search.
            If omitted then only return ``.png``, ``.jpg`` files
        follow_links: If true then follow symbolic links when recursively searching the given dataset directory
        shuffle_index_directory: Path to directory to hold generated index of the dataset
            If omitted, then an index is generated at <directory>/.index
        list_valid_filenames_in_directory_function: This is a custom dataset processing callback.
            It should return a list of valid file names for the given class.
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

        process_samples_function: This allows for processing the samples BEFORE they're returned by this API.
            This allows for adding/removing samples.
            It has the following function signature:

            .. highlight:: python
            .. code-block:: python

                def process_samples(
                    directory:str, # The provided directory to this API
                    sample_paths:Dict[str,str], # A dictionary: <class name>, [<sample paths relative to directory>]
                    split:Tuple[float,float],
                    follow_links:bool,
                    white_list_formats:List[str],
                    shuffle:bool,
                    seed:int,
                    **kwargs
                )
                    ...

    Returns:
        Returns a tuple of two tf.data.Dataset, (samples, label_ids)
    """

    white_list_formats = white_list_formats or ('.png','.jpg')

    sample_paths, sample_labels = list_dataset_directory(
        directory=directory,
        classes=classes,
        max_samples_per_class=max_samples_per_class,
        list_valid_filenames_in_directory_function=list_valid_filenames_in_directory_function,
        shuffle_index_directory=shuffle_index_directory,
        unknown_class_percentage=unknown_class_percentage,
        unknown_class_label='_unknown_',
        split=split,
        shuffle=shuffle,
        seed=seed,
        white_list_formats=white_list_formats,
        return_absolute_paths=True,
        follow_links=follow_links,
        class_counts=class_counts,
        process_samples_function=process_samples_function
    )

    label_ds = tf.data.Dataset.from_tensor_slices(np.array(sample_labels, dtype=np.int32))
    path_ds = tf.data.Dataset.from_tensor_slices(sample_paths)

    if return_image_data:
        feature_ds = path_ds.map(lambda x: read_image_file(x, return_numpy=False, target_channels=0))

    else:
        feature_ds = path_ds

    if onehot_encode:
        label_ds = label_ds.map(
            lambda x: tf.one_hot(x, depth=len(classes), dtype=tf.int32),
        )

    return feature_ds, label_ds


def parallel_process(
    dataset:tf.data.Dataset,
    callback:Callable,
    dtype:Union[np.dtype, Tuple[np.dtype]]=np.float32,
    n_jobs:Union[int,float]=4,
    job_batch_size:int=None,
    pool:ProcessPool=None,
    name='ParallelProcess',
    env:Dict[str,str]=None,
    disable_gpu_in_subprocesses=True,
) -> Tuple[tf.data.Dataset, ProcessPool]:
    """Parallel process the dataset

    This will invoke the given ``callback``
    across the available CPUs in the system which can greatly improve throughput.

    .. note::
        This uses the `tf.numpy_function <https://www.tensorflow.org/api_docs/python/tf/numpy_function>`_
        API which can slow processing in some instances.

    Args:
        dataset: The Tensorflow dataset to parallel process
        callback: The callback to invoke in parallel processes
            This callback must be at the root of the python module (i.e. it cannot be nested or a class method)
        dtype: The data type that the ``callback`` returns,
            this can also be a list of dtypes if the callback returns multiple np.ndarrays
        n_jobs: The number of jobs (i.e. CPU cores) to use for processing
            This can either be an integer or a float between (0,1.0]
        job_batch_size: This size of the batches to use for processing
            If omitted, then use the calculated ``n_jobs``
        pool: An existing processing pool. If omitted then create a new pool
        name: The prefix to use in the model graph
        env: Optional OS environment variables to export in the parallel subprocesses
        disable_gpu_in_subprocesses: By default the GPU is disabled in the parallel subprocesses

    Returns:
        (tf.data.Dataset, ProcessPool),
        a tuple of the updated dataset with the parallel processing and the associated process pool

    """
    n_jobs = calculate_n_jobs(n_jobs)
    job_batch_size = job_batch_size or n_jobs

    process_pool = pool or ProcessPool(
        entry_point=callback,
        n_jobs=n_jobs,
        name=name,
        env=env,
        disable_gpu_in_subprocesses=disable_gpu_in_subprocesses,
        logger=get_mltk_logger()
    )

    def _np_parallel_process(*args):
        pool_batch = process_pool.create_batch(len(args[0]))

        if len(args) == 1:
            for x in args[0]:
                process_pool(x, pool_batch=pool_batch)
        else:
            for i in range(pool_batch.size):
                batch_x = []
                for a in args:
                    batch_x.append(a[i])
                process_pool(*batch_x, pool_batch=pool_batch)

        results = pool_batch.wait()
        if isinstance(dtype, (list,tuple)):
            retval = []
            r0 = results[0]
            for i, t in enumerate(dtype):
                r0_i = r0[i]
                r0_i_shape = (pool_batch.size,) + r0_i.shape
                retval.append(np.empty(r0_i_shape, dtype=t))

            for batch_i, r in enumerate(results):
                for r_i, x in enumerate(r):
                    retval[r_i][batch_i] = x
            return retval

        else:
            return np.array(results, dtype=dtype)

    @tf.function#(autograph=False)
    def _tf_parallel_process(*args):
        return tf.numpy_function(
            _np_parallel_process,
            args,
            dtype
        )

    ds = dataset.batch(job_batch_size)
    ds = ds.map(_tf_parallel_process)
    ds = ds.unbatch()


    return ds, process_pool


def enable_numpy_behavior() -> bool:
    """Enable NumPy behavior on Tensors.

    NOTE: This requires Tensorflow 2.5+

    Enabling NumPy behavior has three effects:

    - It adds to tf.Tensor some common NumPy methods such as T, reshape and ravel.
    - It changes dtype promotion in tf.Tensor operators to be compatible with NumPy. For example, tf.ones([], tf.int32) + tf.ones([], tf.float32) used to throw a "dtype incompatible" error, but after this it will return a float64 tensor (obeying NumPy's promotion rules).
    - It enhances tf.Tensor's indexing capability to be on par with NumPy's.

    Refer to the `Tensorflow docs <https://tensorflow.org/versions/r2.5/api_docs/python/tf/experimental/numpy/experimental_enable_numpy_behavior>`_ for more details.

    Returns:
        True if the numpy behavior was enabled, False else
    """
    enabled_numpy_behavior = globals().get('enabled_numpy_behavior', None)

    if enabled_numpy_behavior is None:
        try:
            enabled_numpy_behavior = True
            from tensorflow.python.ops.numpy_ops import np_config
            np_config.enable_numpy_behavior()
        except:
            try:
                tf.experimental.numpy.experimental_enable_numpy_behavior()
            except:
                enabled_numpy_behavior = False

        globals()['enabled_numpy_behavior'] = enabled_numpy_behavior

    return enabled_numpy_behavior



def _update_silence_samples(sample_paths:List[str], sample_rate_hz:int):
    """This will create a "silence" audio sample and add it to the dataset where necessary"""
    if sample_rate_hz == -1 or not sample_rate_hz:
        sample_rate_hz = 16000
    silence_wav_path = create_tempdir('temp') + f'/silence_{sample_rate_hz}hz.wav'
    if not os.path.exists(silence_wav_path):
        with wave.open(silence_wav_path, mode='wb') as wav:
            nchannels = 1
            nframes = sample_rate_hz * 1
            comptype = 'NONE'
            compname = 'not compressed'
            wav.setparams((nchannels, 2, sample_rate_hz, nframes, comptype, compname))
            wav.writeframes(bytearray([0] * nframes * 2))


    for i, p in enumerate(sample_paths):
        if p is None:
            sample_paths[i] = silence_wav_path



