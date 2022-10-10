import os
from typing import Tuple, List, Dict
import copy
import random
import math
import multiprocessing
import numpy as np

from mltk.core import get_mltk_logger






def list_dataset_directory(
    directory:str,
    classes:List[str],
    unknown_class_percentage=1.0,
    unknown_class_label=None,
    empty_class_percentage=1.0,
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
        unknown_class_label: Class label to be considered "unknown". See the ``classes`` arg for more details
        empty_class_percentage: If a ``empty_class_label`` class is added to the class list, then 'silence' class samples will automatically
            be added to batches. This specifies the percentage of of samples to generate relative the smallest number
            of samples of other classes. For instance, if another class has 1000 samples and empty_class_percentage=0.8,
            then the number of 'empty' class samples generated will be 800.
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
                        shuffle_index_directory:str
                ) -> Tuple[str, List[str]]
                    ...
                    return search_class, filenames

    Returns:
        Returns a tuple of two lists, (samples paths, label_ids)
    """
    sample_paths = []
    sample_class_ids = []
    classes = copy.deepcopy(classes)
    if class_counts is None:
        class_counts = {}
    add_unknown_class = unknown_class_label and unknown_class_label in classes
    add_empty_class = empty_class_label and empty_class_label in classes
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

        class_count = len(filenames)
        class_counts[class_label] = class_count
        sample_class_ids.extend([classes.index(class_label)] * class_count)
        sample_paths.extend(filenames)
    
    # Determine the average number of samples for a given classes
    avg_sample_count = sum(class_counts.values()) // len(class_counts)
    
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
                all_unknown_filenames[class_label] = filenames
            
            
        # NOTE: It's best effort to get up to unknown_sample_count 
        unknown_sample_count = max(int(avg_sample_count * unknown_class_percentage), 1)
        unknown_filenames = []
        count_per_class  = max(unknown_sample_count // len(all_unknown_filenames), 1)
        for filenames in all_unknown_filenames.values():
            chunk_len = min(len(filenames), count_per_class)
            unknown_filenames.extend(filenames[:chunk_len])
            if len(unknown_filenames) >= unknown_sample_count:
                break

        class_count = len(unknown_filenames)
        class_counts[unknown_class_label] = class_count
        sample_class_ids.extend([classes.index(unknown_class_label)] * class_count)
        sample_paths.extend(unknown_filenames)

    if add_empty_class:
        empty_class_count = max(1, int(avg_sample_count * empty_class_percentage))
        class_counts[empty_class_label] = empty_class_count
        sample_class_ids.extend([classes.index(empty_class_label)] * empty_class_count)
        sample_paths.extend([None] * empty_class_count)


    pool.close()
    pool.join()

    errs = []
    for clazz, n_sample in class_counts.items():
        if n_sample == 0:
            errs.append(f'No samples found for class: {clazz}')
    if errs:
        raise RuntimeError('\n'.join(errs))


    if shuffle:
        shuffle_seed = seed or 43
        rng = np.random.RandomState(shuffle_seed)
        rng.shuffle(sample_paths)
        rng = np.random.RandomState(shuffle_seed)
        rng.shuffle(sample_class_ids)

    if return_absolute_paths:
        sample_paths = [f'{directory}/{fn}' if fn else fn for fn in sample_paths]

    return sample_paths, sample_class_ids



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
        class_base_dir = f'{base_directory}/{search_class}'
        for root, _, files in os.walk(base_directory, followlinks=follow_links):
            root = root.replace('\\', '/')
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