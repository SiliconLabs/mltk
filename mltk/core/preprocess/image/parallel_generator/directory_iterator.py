"""Utilities for real-time data augmentation on image data.
"""
import os
import copy
import math
import random
from typing import List, Tuple, Dict

import numpy as np


from mltk.core.utils import get_mltk_logger
from mltk.core.preprocess.utils import list_dataset_directory
from .iterator import ParallelProcessParams, ParallelIterator



class ParallelDirectoryIterator(ParallelIterator):
    """Iterator capable of reading images from a directory on disk.

    # Arguments
        directory: string, path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"rgba"`, `"grayscale"`.
            Color mode to read images.
        classes: Optional list of strings, names of subdirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input images are yielded).
        batch_shape: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
            If set to False, sorts the data in alphanumeric order.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        follow_links: boolean,follow symbolic links to subdirectories
        subset: Subset of data (`"training"` or `"validation"`) if
            validation_split is set in ImageDataGenerator.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
        dtype: Dtype to use for generated arrays.
    """
    allowed_class_modes = {'categorical', 'binary', 'sparse', 'input', None}

    def __init__(
        self,
        directory,
        image_data_generator,
        target_size=(256, 256),
        color_mode='rgb',
        classes=None,
        class_mode='categorical',
        batch_shape=None,
        shuffle=True,
        seed=None,
        data_format='channels_last',
        save_to_dir=None,
        save_prefix='',
        save_format='png',
        follow_links=False,
        subset=None,
        interpolation='nearest',
        dtype='float32',
        cores=0.25,
        debug=False,
        max_batches_pending=4,
        get_batch_function=None,
        preprocessing_function=None,
        noaug_preprocessing_function=None,
        list_valid_filenames_in_directory_function=None,
        shuffle_index_dir=None,
        max_samples_per_class=-1,
        disable_gpu_in_subprocesses=True,
        class_counts:Dict[str,int]=None,
    ):

        self.directory = directory
        self.cores = cores
        self.class_labels = copy.deepcopy(classes)
        self.debug = debug
        self.max_batches_pending = max_batches_pending
        self.disable_gpu_in_subprocesses = disable_gpu_in_subprocesses

        
        if class_mode not in self.allowed_class_modes:
            raise ValueError('Invalid class_mode: {}; expected one of: {}'
                             .format(class_mode, self.allowed_class_modes))
        self.class_mode = class_mode
        self.dtype = dtype
        # First, count the number of samples and classes.
        self.samples = 0

        if not classes:
            if shuffle_index_dir:
                raise Exception('Must specify classes if shuffle_index_dir is not None')
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
            self.class_labels = classes
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))


        process_params = ParallelProcessParams(
            image_data_generator=image_data_generator,
            target_size= target_size,
            batch_shape=batch_shape,
            color_mode=color_mode,
            data_format=data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            subset=subset,
            interpolation=interpolation,
            class_indices=self.class_indices,
            dtype=dtype,
            directory=directory,
            class_mode=class_mode,
            get_batch_function=get_batch_function,
            noaug_preprocessing_function=noaug_preprocessing_function,
            preprocessing_function=preprocessing_function
        )

        sample_paths, sample_class_ids = list_dataset_directory(
            directory=directory,
            classes=classes,
            max_samples_per_class=max_samples_per_class,
            list_valid_filenames_in_directory_function=list_valid_filenames_in_directory_function,
            unknown_class_label=None, 
            empty_class_label=None,
            shuffle_index_directory=shuffle_index_dir,
            split=process_params.split,
            seed=seed,
            white_list_formats=self.white_list_formats,
            return_absolute_paths=False,
            follow_links=follow_links,
            class_counts=class_counts
        )

        self.filenames = sample_paths
        self.samples = len(self.filenames)
        self.classes = np.array(sample_class_ids, dtype=np.int32)
        
        self._filepaths = []
        for fname in self.filenames:
            if isinstance(fname, (list,tuple)):
                self._filepaths.append(tuple(f'{self.directory}/{fn}' for fn in fname))
            else:
                self._filepaths.append(f'{self.directory}/{fname}')

        if batch_shape[0] == -1:
            batch_shape = (self.samples,) + batch_shape[1:]

        super(ParallelDirectoryIterator, self).__init__(
            self.samples,
            batch_shape,
            shuffle,
            seed,
            process_params=process_params
        )

    @property
    def filepaths(self):
        return self._filepaths

    @property
    def sample_count(self):
        return len(self) * self.batch_size

    @property
    def labels(self):
        return self.classes[:self.sample_count]


    @property  # mixin needs this property to work
    def sample_weight(self):
        # no sample weights will be returned
        return None

