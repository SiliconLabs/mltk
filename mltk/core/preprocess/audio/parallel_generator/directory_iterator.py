"""Utilities for real-time data augmentation on image data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict
import copy
import numpy as np
from mltk.core.preprocess.utils import list_dataset_directory
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
        add_channel_dimension=True,
        class_counts:Dict[str,int]=None
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

        sample_paths, sample_class_ids = list_dataset_directory(
            directory=directory,
            classes=classes,
            max_samples_per_class=max_samples_per_class,
            list_valid_filenames_in_directory_function=list_valid_filenames_in_directory_function,
            shuffle_index_directory=shuffle_index_dir,
            unknown_class_label='_unknown_',
            unknown_class_percentage=unknown_class_percentage,
            empty_class_label='_silence_',
            empty_class_percentage=silence_class_percentage,
            split=process_params.split,
            seed=seed,
            white_list_formats=self.white_list_formats,
            return_absolute_paths=False,
            follow_links=follow_links,
            class_counts=class_counts
        )
        
        self.filenames = sample_paths
        self.samples = len(self.filenames)
        self.num_classes = len(classes)
        self.classes = np.array(sample_class_ids, dtype=np.int32)

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
