"""Utilities for real-time data augmentation on image data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import math
import multiprocessing.pool
import random
import numpy as np
from typing import List, Tuple

from mltk.core.utils import get_mltk_logger

from .iterator import ParallelProcessParams, ParallelIterator



class ParallelDirectoryIterator(ParallelIterator):
    allowed_class_modes = {'categorical', 'binary', 'sparse', 'input', None}

    def __init__(
        self,
        directory,
        audio_data_generator,
        classes,
        shuffle_index_dir,
        sample_shape,
        class_mode='categorical',
        batch_size=32,
        sample_rate=16000,
        sample_length_ms=1000,
        unknown_class_percentage=0.8,
        silence_class_percentage=0.6,
        shuffle=True,
        seed=None,
        save_to_dir=None,
        save_prefix='',
        save_format='png',
        follow_links=False,
        subset=None,
        dtype='float32',
        frontend_dtype=None,
        cores=0.25,
        debug=False,
        max_batches_pending=4,
        get_batch_function=None,
        noaug_preprocessing_function=None,
        preprocessing_function=None,
        postprocessing_function=None,
        list_valid_filenames_in_directory_function=None,
        max_samples_per_class=-1,
        frontend_enabled=True,
        disable_gpu_in_subprocesses=True,
        add_channel_dimension=True
    ):

        self.directory = directory
        classes = copy.deepcopy(classes)
        self.max_batches_pending = max_batches_pending
        self.cores = cores
        self.debug = debug
        self.disable_gpu_in_subprocesses = disable_gpu_in_subprocesses

        
        if class_mode not in self.allowed_class_modes:
            raise ValueError('Invalid class_mode: {}; expected one of: {}'
                             .format(class_mode, self.allowed_class_modes))
        self.class_mode = class_mode
        self.dtype = dtype
        # First, count the number of samples and classes.
        self.samples = 0
        
        add_unknown_class = '_unknown_' in classes
        add_silence_class = '_silence_' in classes
        
        if add_unknown_class:
            unknown_classes = _find_unknown_clases(classes, directory)
            if len(unknown_classes) == 0:
                raise Exception("Failed to find 'unknown' classes in {}".format(directory))

        self.class_indices = dict(zip(classes, range(len(classes))))


        process_params = ParallelProcessParams(
            audio_data_generator=audio_data_generator,
            sample_rate=sample_rate,
            sample_length_ms=sample_length_ms,
            sample_shape=sample_shape,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            subset=subset,
            class_indices=self.class_indices,
            dtype=dtype,
            frontend_dtype=frontend_dtype,
            directory=directory,
            class_mode=class_mode,
            get_batch_function=get_batch_function,
            noaug_preprocessing_function=noaug_preprocessing_function,
            preprocessing_function=preprocessing_function,
            postprocessing_function=postprocessing_function,
            frontend_enabled=frontend_enabled,
            add_channel_dimension=add_channel_dimension
        )



        pool = multiprocessing.pool.ThreadPool()

        # Second, build an index of the images
        # in the different class subfolders.
        results = []
        self.filenames = []
        list_valid_filenames_in_directory_function = list_valid_filenames_in_directory_function or _list_valid_filenames_in_directory

        for clazz in classes:
            if clazz in ['_silence_', '_unknown_']:
                continue 
            results.append(
            pool.apply_async(list_valid_filenames_in_directory_function, (
                directory, 
                clazz, 
                self.white_list_formats, 
                process_params.split, 
                follow_links, 
                shuffle_index_dir
            )))
        
        class_counts = {}
        classes_list = []
        for res in results:
            class_label, filenames = res.get()

            if max_samples_per_class != -1:
                max_len = min(max_samples_per_class, len(filenames))
                filenames = filenames[:max_len]

            class_count = len(filenames)
            class_counts[class_label] = class_count
            classes_list.append([self.class_indices[class_label]] * class_count)
            self.filenames += filenames
        
        
        # Determine the average number of samples for a given classes
        avg_sample_count = sum(class_counts.values()) // len(class_counts)
        
        # Add the 'silence' class if necessary
        if add_silence_class:
            silence_sample_count = max(int(avg_sample_count * silence_class_percentage), 1)
            class_counts['_silence_'] = silence_sample_count
            classes_list.append([self.class_indices['_silence_']] * silence_sample_count)
            self.filenames += [None] * silence_sample_count
            
        # Add the 'unknown' class if necessary
        if add_unknown_class:
            results = []
            for clazz in unknown_classes:
                results.append(
                pool.apply_async(list_valid_filenames_in_directory_function, (
                    directory, 
                    clazz, 
                    self.white_list_formats, 
                    process_params.split, 
                    follow_links, 
                    shuffle_index_dir
                )))
                
            all_unknown_filesnames = {}
            for res in results:
                class_label, filenames = res.get()
                
                if len(filenames) > 0:
                    all_unknown_filesnames[class_label] = filenames
                
                
            # NOTE: It's best effort to get up to unknown_sample_count 
            unknown_sample_count = max(int(avg_sample_count * unknown_class_percentage), 1)
            unknown_filesnames = []
            count_per_class  = max(unknown_sample_count // len(all_unknown_filesnames), 1)
            for filenames in all_unknown_filesnames.values():
                chunk_len = min(len(filenames), count_per_class)
                unknown_filesnames += filenames[:chunk_len]
                if len(unknown_filesnames) >= unknown_sample_count:
                    break

            class_count = len(unknown_filesnames)
            class_counts['_unknown_'] = class_count
            classes_list.append([self.class_indices['_unknown_']] * class_count)
            self.filenames += unknown_filesnames

        pool.close()
        pool.join()

        errs = []
        for clazz, n_sample in class_counts.items():
            if n_sample == 0:
                errs.append(f'No samples found for class: {clazz}')
        if errs:
            raise Exception('\n'.join(errs))

        
        self.samples = len(self.filenames)
        self.num_classes = len(class_counts)
        
        i = 0
        self.classes = np.zeros((self.samples,), dtype='int32')
        for classes in classes_list:
            self.classes[i:i + len(classes)] = classes
            i += len(classes)

        if batch_size == -1:
            batch_size = self.samples


        super(ParallelDirectoryIterator, self).__init__(
            self.samples,
            batch_size,
            shuffle,
            seed,
            process_params=process_params
        )



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


def _find_unknown_clases(known_classes, base_directory):
    unknown_classes = []
    
    for class_name in os.listdir(base_directory):
        if not os.path.isdir(os.path.join(base_directory, class_name)):
            continue
        if class_name in known_classes or class_name.startswith(('_', '~', '.')):
            continue 
        unknown_classes.append(class_name)
        
    return unknown_classes
    

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

