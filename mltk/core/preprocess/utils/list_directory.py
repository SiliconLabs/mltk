"""Utilities for listing dataset directories"""

import os
from typing import Tuple, List, Dict, Callable
import copy
import random
import math
import multiprocessing
import numpy as np
from numpy.random import RandomState

from mltk.core import get_mltk_logger
from mltk.utils.python import prepend_exception_msg





def list_dataset_directory(
    directory:str,
    classes:List[str],
    unknown_class_percentage:float=1.0,
    unknown_class_label=None,
    empty_class_percentage:float=1.0,
    empty_class_label:str=None,
    class_counts:Dict[str,int] = None,
    shuffle=False,
    seed=None,
    split:Tuple[float,float]=None,
    max_samples_per_class=-1,
    white_list_formats:List[str]=None,
    follow_links=False,
    shuffle_index_directory:str=None,
    return_absolute_paths:bool=False,
    list_valid_filenames_in_directory_function=None,
    process_samples_function=None,
) -> Tuple[List[str], List[int]]:
    """Load a directory of samples and return a tuple of lists (sample paths, label_ids)

    The given directory should have the structure::

        <class1>/sample1
        <class1>/sample2
        ...
        <class1>/optional sub directory/sample9
        <class2>/sample1
        <class2>/sample2
        ...
        <class3>/sample1
        <class3>/sample2

    Where each <class> is found in the given ``classes`` argument.

    Args:
        directory: Directory path to audio dataset
        classes: List of class labels to include in the returned dataset

            * If ``unknown_class_label`` is added as an entry to the ``classes``, then this API will automatically
              add an 'unknown' class to the generated batches.
              Unused classes in the dataset directory will be randomly selected and used as an 'unknown' class.
              Use the ``unknown_class_percentage`` setting to control the size of this class.
            * If ``empty_class_label`` is added as an entry to the ``classes``, then this API will automatically
              add 'empty' samples with all zeros.
              Use the ``empty_class_percentage`` setting to control the size of this class.

        unknown_class_percentage: If an ``unknown_class_label`` class is added to the class list, then 'unknown' class samples will automatically
            be added to batches. This specifies the percentage of of samples to generate relative the smallest number
            of samples of other classes. For instance, if another class has 1000 samples and unknown_class_percentage=0.8,
            then the number of 'unknown' class samples generated will be 800.
            Set this parameter to None to disable this feature
        unknown_class_label: Class label to be considered "unknown". See the ``classes`` arg for more details
        empty_class_percentage: If a ``empty_class_label`` class is added to the class list, then 'silence' class samples will automatically
            be added to batches. This specifies the percentage of of samples to generate relative the smallest number
            of samples of other classes. For instance, if another class has 1000 samples and empty_class_percentage=0.8,
            then the number of 'empty' class samples generated will be 800.
            Set this parameter to None to disable this feature
        empty_class_label: Class label to be considered "empty". See the ``classes`` arg for more details
        class_counts: Dictionary which will be populated with the sample counts for each class
        shuffle: If true, then shuffle the dataset
        seed: The seed to use for shuffling the dataset
        split: A tuple indicating the (start,stop) percentage of the dataset to return,
            e.g. (.75, 1.0) -> return last 25% of dataset
            If omitted then return the entire dataset
        max_samples_per_class: Maximum number of samples to return per class, this can be useful for debug to limit the dataset size
        return_audio_data: If true then the audio file data is returned, if false then only the audio file path is returned
        white_list_formats: List of file extensions to include in the search.
        follow_links: If true then follow symbolic links when recursively searching the given dataset directory
        shuffle_index_directory: Path to directory to hold generated index of the dataset
            If omitted, then an index is generated at <directory>/.index
        return_absolute_paths: If true then return absolute paths to samples, if false then paths are relative to the given directory
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
                    shuffle_index_directory:str,
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
        Returns a tuple of two lists, (samples paths, label_ids)
    """
    sample_paths = {key: [] for key in classes}
    shuffle_seed = seed or 43
    classes = copy.deepcopy(classes)
    add_unknown_class = unknown_class_percentage and unknown_class_label in classes
    add_empty_class = empty_class_percentage and empty_class_label in classes
    list_valid_filenames_in_directory_function = \
        list_valid_filenames_in_directory_function or list_valid_filenames_in_directory

    if seed:
        random.seed(seed)

    if add_unknown_class:
        unknown_classes = _find_unknown_classes(classes, directory)
        if len(unknown_classes) == 0:
            raise Exception(f"Failed to find 'unknown' classes in {directory}")


    thread_count = min(multiprocessing.cpu_count(), len(classes))
    pool = multiprocessing.pool.ThreadPool(processes=thread_count)

    # Second, build an index of the images
    # in the different class subfolders.
    results = []
    for clazz in classes:
        if clazz == unknown_class_label or clazz == empty_class_label:
            continue
        results.append(
        pool.apply_async(list_valid_filenames_in_directory_function,
            kwds=dict(
                base_directory=directory,
                search_class=clazz,
                white_list_formats=white_list_formats,
                split=split,
                shuffle_index_directory=shuffle_index_directory,
                follow_links=follow_links,
        )))


    for res in results:
        class_label, filenames = res.get()
        if max_samples_per_class != -1:
            max_len = min(max_samples_per_class, len(filenames))
            filenames = filenames[:max_len]

        sample_paths[class_label] = filenames

    if len(sample_paths) == 0:
        raise RuntimeError(f'No classes found for {", ".join(classes)} in {directory}')

    # Determine the average number of samples for a given classes (that aren't "unknown" or "empty")
    n_classes = len(classes)
    if add_unknown_class:
        n_classes -= 1
    if add_empty_class:
        n_classes -= 1
    avg_sample_count = sum([len(x) for x in sample_paths.values()]) // n_classes

    # Add the 'unknown' class if necessary
    if add_unknown_class:
        results = []
        for clazz in unknown_classes:
            results.append(
            pool.apply_async(list_valid_filenames_in_directory_function,
                kwds=dict(
                    base_directory=directory,
                    search_class=clazz,
                    white_list_formats=white_list_formats,
                    split=split,
                    shuffle_index_directory=shuffle_index_directory,
                    follow_links=follow_links,
            )))

        all_unknown_filenames = {}
        for res in results:
            class_label, filenames = res.get()
            if len(filenames) > 0:
                all_unknown_filenames[class_label] = list(filenames)


        # NOTE: It's best effort to get up to unknown_sample_count
        unknown_sample_count = max(int(avg_sample_count * unknown_class_percentage), 1)

        unknown_filenames = []
        count_per_class  = max(unknown_sample_count // len(all_unknown_filenames), 1)
        for filenames in all_unknown_filenames.values():
            rng = RandomState(shuffle_seed)
            rng.shuffle(filenames)
            chunk_len = min(len(filenames), count_per_class)
            unknown_filenames.extend(filenames[:chunk_len])
            if len(unknown_filenames) >= unknown_sample_count:
                break

        sample_paths[unknown_class_label] = unknown_filenames


    if add_empty_class:
        empty_class_count = max(1, int(avg_sample_count * empty_class_percentage))
        sample_paths[empty_class_label] = [None] * empty_class_count

    pool.close()
    pool.join()

    if process_samples_function is not None:
        try:
            process_samples_function(
                directory=directory,
                sample_paths=sample_paths,
                split=split,
                follow_links=follow_links,
                white_list_formats=white_list_formats,
                shuffle=shuffle,
                seed=shuffle_seed
            )
        except Exception as e:
            prepend_exception_msg(e, 'Error in process_samples_function callback')
            raise

    # If the API was provided with a "class_counts" dictionary,
    # then we want to populate the given dictionary.
    # Otherwise, just populate a dummy,local dictionary
    if class_counts is None:
        class_counts = {}
    class_counts.update({key: len(samples) for (key,samples) in sample_paths.items()})

    errs = []
    for clazz, n_sample in class_counts.items():
        if n_sample == 0:
            errs.append(f'No samples found for class: {clazz}')
    if errs:
        raise RuntimeError('\n'.join(errs))


    if max_samples_per_class != -1:
        for clazz, paths in sample_paths.items():
            max_len = min(max_samples_per_class, len(paths))
            sample_paths[clazz] = paths[:max_len]

    sample_path_list = []
    sample_class_id_list = []
    for key, paths in sample_paths.items():
        sample_path_list.extend(paths)
        sample_class_id_list.extend([classes.index(key)] * len(paths))

    if shuffle:
        rng = RandomState(shuffle_seed)
        rng.shuffle(sample_path_list)
        rng = RandomState(shuffle_seed)
        rng.shuffle(sample_class_id_list)

    if return_absolute_paths:
        sample_path_list = [f'{directory}/{fn}' if fn else fn for fn in sample_path_list]

    return sample_path_list, sample_class_id_list



def list_valid_filenames_in_directory(
    base_directory:str,
    search_class:str,
    white_list_formats:List[str]=None,
    split:Tuple[float,float]=None,
    follow_links:bool=False,
    shuffle_index_directory:str=None
) -> Tuple[str, List[str]]:
    """File all files in the search directory for the specified class

    if shuffle_index_directory is None:
        then sort the filenames alphabetically and save to the list file:
        <base_directory>/.index/<search_class>.txt
    else:
        then randomly shuffle the files and save to the list file:
        <shuffle_index_directory>/.index/<search_class>.txt

    Args:
        base_directory: Search directory for the current class
        search_class: Label of class to search
        white_list_formats: List of file extensions to search
        split: A tuple indicating the (start,stop) percentage of the dataset to return,
            e.g. (.75, 1.0) -> return last 25% of dataset
            If omitted then return the entire dataset
        follow_links: If true then follow symbolic links when recursively searching the given dataset directory
        shuffle_index_directory: Path to directory to hold generated index of the dataset
    Returns:
        (search_class, list(relative paths),
        a tuple of the given ``search_class`` and list of file paths relative to the ``base_directory``
    """
    file_list = []
    base_directory = base_directory.replace('\\', '/')

    if shuffle_index_directory is None:
        index_path = f'{base_directory}/.index/{search_class}.txt'
    else:
        index_path = f'{shuffle_index_directory}/.index/{search_class}.txt'

    if isinstance(white_list_formats, list):
        white_list_formats = tuple(white_list_formats)

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
        class_base_dir = f'{base_directory}/{search_class}/'
        for root, _, files in os.walk(base_directory, followlinks=follow_links):
            root = root.replace('\\', '/') + '/'
            if not root.startswith(class_base_dir):
                continue

            for fname in files:
                if white_list_formats and not fname.lower().endswith(white_list_formats):
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


    filenames = split_file_list(
        paths=file_list,
        split=split
    )

    return search_class, filenames


def split_file_list(
    paths:List[str],
    split:Tuple[float,float]=None
) -> List[str]:
    """Split list of file paths

    Args:
        paths: List of file paths
        split: A tuple indicating the (start,stop) percentage of the dataset to return,
            e.g. (.75, 1.0) -> return last 25% of dataset
            If omitted then return the entire dataset

    Return:
        Split file paths
    """
    if split:
        num_files = len(paths)
        if split[0] == 0:
            start = 0
            stop = math.ceil(split[1] * num_files)
        else:
            start = math.ceil(split[0] * num_files)
            stop = num_files

        paths = paths[start:stop]

    return paths


def shuffle_file_list_by_group(
    paths:List[str],
    group_callback:Callable[[str], str],
    seed:int=42,
) -> List[str]:
    """Shuffle the given file list by group

    This uses the given 'group_callback' argument to determine the "group"
    that each file path in the given list belongs.
    It then shuffles each group and returns the shuffles groups as a flat list.

    This is useful as it allows for splitting the list into training and validation
    subsets while ensuring that the same group does not appear in both subsets.

    Args:
        paths: List of file paths
        group_callback: Callback that takes an element of the given 'paths' array and returns its corresponding "group"
        seed: Optional seed used to do that random shuffle
    Return:
        Shuffle list of groups of files
    """
    # Group all the paths by given group_id
    file_groups = {}
    for p in sorted(paths):
        group_id = group_callback(p)
        if group_id not in file_groups:
            file_groups[group_id] = []
        file_groups[group_id].append(p)

    # Shuffle the group_ids
    rng = RandomState(seed)
    group_ids = sorted(file_groups.keys())
    rng.shuffle(group_ids)

    # Flatten the shuffled groups
    file_list = []
    for group_id in group_ids:
        file_list.extend(file_groups[group_id])

    return file_list


def _find_unknown_classes(
    known_classes:List[str],
    base_directory:str
) -> List[str]:
    """Return a list of dataset class names not in the given known_classes"""
    unknown_classes = []

    for class_name in os.listdir(base_directory):
        if not os.path.isdir(os.path.join(base_directory, class_name)):
            continue
        if class_name in known_classes or class_name.startswith(('_', '~', '.')):
            continue
        unknown_classes.append(class_name)

    return unknown_classes