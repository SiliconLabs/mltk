"""Utilities for real-time data augmentation on image data.
"""
import os
import copy
import math
import multiprocessing.pool
import random
from typing import List, Tuple

import numpy as np

from keras_preprocessing.image.utils import _list_valid_filenames_in_directory

from mltk.core.utils import get_mltk_logger
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
        disable_gpu_in_subprocesses=True
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

        pool = multiprocessing.pool.ThreadPool(processes=min(4, self.num_classes))

        # Second, build an index of the images
        # in the different class subfolders.
        results = []
        self.filenames = []
        list_valid_filenames_in_directory_function = list_valid_filenames_in_directory_function or _list_valid_filenames_in_directory

        for clazz in classes:
            results.append(
            pool.apply_async(list_valid_filenames_in_directory_function, (
                directory, 
                clazz, 
                self.white_list_formats, 
                process_params.split,
                follow_links, 
                shuffle_index_dir
            )))

        errs = []
        classes_list = []
        for res in results:
            class_label, filenames = res.get()
            if len(filenames) == 0:
                errs.append(f'No samples found for class: {class_label}')

            if max_samples_per_class != -1:
                max_len = min(max_samples_per_class, len(filenames))
                filenames = filenames[:max_len]
            
            classes_list.append([self.class_indices[class_label]] * len(filenames))
            self.filenames += filenames
        
        if errs:
            raise Exception('\n'.join(errs))

        self.samples = len(self.filenames)
        
        i = 0
        self.classes = np.zeros((self.samples,), dtype='int32')
        for class_ids in classes_list:
            self.classes[i:i + len(class_ids)] = class_ids
            i += len(class_ids)

        pool.close()
        pool.join()
        
        
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



def _list_valid_filenames_in_directory(
    base_directory:str, 
    search_class:str, 
    white_list_formats:List[str], 
    split:float, 
    follow_links:bool, 
    shuffle_index_directory:str
) -> Tuple[str, List[str]]:
    """File all files in the search directory for the specified class

    if shuffle_index_directory is None:
        then sort the filenames alphabetically and save to the list file:
        <base_directory>/.index/<search_class>.txt
    else:
        then randomly shuffle the files and save to the list file:
        <shuffle_index_directory>/.index/<search_class>.txt

    """
    file_list = []
    base_directory = base_directory.replace('\\', '/')

    if shuffle_index_directory is None:
        index_path = f'{base_directory}/.index/{search_class}.txt'
    else:
        index_path = f'{shuffle_index_directory}/.index/{search_class}.txt'


    # If the index file exists, then read it
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            for line in f:
                filename = line.strip()
                filepath = f'{base_directory}/{filename}'
                if not os.path.exists(filepath):
                    get_mltk_logger().warning(f'File {filepath} not found in existing index, re-generating index')
                    file_list = []
                    break
                file_list.append(filename)


    if len(file_list) == 0:
        get_mltk_logger().info(f'Generating index: {index_path} ...')
        # Else find all files for the given class in the search directory
        # NOTE: The dataset directory structure should be:
        # <dataset base dir>/<class1>/
        # <dataset base dir>/<class1>/sample1.jpg
        # <dataset base dir>/<class1>/sample2.jpg
        # <dataset base dir>/<class1>/subfolder1/sample3.jpg
        # <dataset base dir>/<class1>/subfolder2/sample4.jpg
        # <dataset base dir>/<class2>/...
        # <dataset base dir>/<class3>/...
        #
        # This will recursively return all sample files under <dataset base dir>/<class x>
        class_base_dir = f'{base_directory}/{search_class}'
        for root, _, files in os.walk(base_directory, followlinks=follow_links):
            root = root.replace('\\', '/')
            if not root.startswith(class_base_dir):
                continue
            
            for fname in files:
                if not fname.lower().endswith(white_list_formats):
                    continue
                abs_path = os.path.join(root, fname)
                rel_path = os.path.relpath(abs_path, base_directory)
                file_list.append(rel_path.replace('\\', '/'))


        # Randomly shuffle the list if necessary
        if shuffle_index_directory is not None:
            random.shuffle(file_list)
        else:
            # Otherwise sort it alphabetically
            file_list = sorted(file_list)

        # Write the file list file
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        with open(index_path, 'w') as f:
            for p in file_list:
                f.write(p + '\n')


    if split:
        num_files = len(file_list)
        if split[0] == 0:
            start = 0
            stop = math.ceil(split[1] * num_files)
        else:
            start = math.ceil(split[0] * num_files)
            stop = num_files
            
        filenames = file_list[start:stop] 
    
    else:
        filenames = file_list

    return search_class, filenames

