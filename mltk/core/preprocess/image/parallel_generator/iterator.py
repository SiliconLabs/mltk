"""Utilities for real-time data augmentation on image data.
"""
import os
import sys
import time
import threading
import inspect
import queue
from typing import List, Tuple
import random
import numpy as np

from mltk.core.keras import (
    array_to_img,
    img_to_array,
    load_img
)
from mltk.core import get_mltk_logger
from mltk.core.keras import DataSequence
from mltk.utils.process_pool import ProcessPool, calculate_n_jobs


class ParallelIterator(DataSequence):
    """Base class for image data iterators.

    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_shape: Integer, shape of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """
    white_list_formats = ('png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff', 'npy')

    def __init__(self, n, batch_shape, shuffle, seed, process_params):
        super().__init__()

        self.n = n
        self.batch_shape = batch_shape
        self.seed = seed
        self.total_batches_seen = 0
        self.batch_index = 0
        self.shuffle = shuffle
        self.process_params = process_params
        self.check_reuse_batch_zero = True

        n_jobs = calculate_n_jobs(self.cores)
        # Use half the number of jobs for the validation subset
        if process_params.validation_split and process_params.subset == 'validation':
            n_jobs = max(int(n_jobs*.5), 1)

        get_batch_function = self.process_params.get_batch_function or get_batches_of_transformed_samples
        self.pool = ProcessPool(
            name=self.process_params.subset,
            entry_point=get_batch_function,
            n_jobs=n_jobs,
            debug=self.debug,
            disable_gpu_in_subprocesses=self.disable_gpu_in_subprocesses,
            logger=get_mltk_logger()
        )

        self.batch_generation_started = threading.Event()
        self.current_batch_finished = threading.Event()
        self.current_batch_finished.set()

        self.batch_data = BatchData(
            len(self), 
            shuffle,
            pool=self.pool
        )
       
        self.batch_thread = threading.Thread(
            target=self._generate_batch_data_safe, 
            name=f'Batch data generator:{process_params.subset}',
            daemon=True
        )
        self.batch_thread.start()

 
    @property
    def is_shutdown(self) -> bool:
        return not self.pool.is_running

    @property
    def is_running(self) -> bool:
        return not self.is_shutdown

    @property
    def batch_size(self) -> int:
        return self.batch_shape[0]

        
    def reset(self):
        self.batch_generation_started.clear()
        self.current_batch_finished.set()
        self.batch_data.reset()
        self.batch_index = 0
    
    
    def shutdown(self, wait=True):
        if wait:
            self.current_batch_finished.wait(30)
        self.reset()
        self.pool.shutdown()
       


    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError(
                f'Asked to retrieve element {idx}, but the Sequence has length {len(self)}'
            )

        # Some TF APIs use a "peek_and_restore" operation on the first batch.
        # To account for this, we check the callback stack for a function that has "peek" in it
        # If so, we notify the batch_data.get() API that it should save this batch as it will be re-used
        save_batch_zero = False
        if idx == 0 and self.check_reuse_batch_zero:
            self.check_reuse_batch_zero = False
            callback_function_name = sys._getframe(1).f_code.co_name
            if 'peek' in callback_function_name:
                save_batch_zero = True

        self.batch_generation_started.set()
        self.current_batch_finished.clear()
        
        retval, is_last = self.batch_data.get(idx, save_batch_zero=save_batch_zero)
        if is_last:
            self.current_batch_finished.set()
        return retval


    def __len__(self):
        return (self.n + self.batch_size -1) // self.batch_size  # round up


    def on_epoch_end(self):
        pass


    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        self.batch_index = 0
        return self


    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        if self.batch_index >= len(self):
            # Clear the started flag, but do NOT reset
            # This way we don't waste any processed batch data
            self.batch_generation_started.clear()
            raise StopIteration()
        
        self.batch_generation_started.set()
        self.current_batch_finished.clear()
        retval, is_last = self.batch_data.get(self.batch_index)
        if is_last:
            self.current_batch_finished.set()
        
        self.batch_index += 1
        
        return retval


    def _generate_batch_data_safe(self):
        try:
            self._generate_batch_data()
        except Exception as e:
            if not self.is_shutdown:
                get_mltk_logger().error(f'Exception during batch data processing, err: {e}', exc_info=e)
                return
            self.shutdown(wait=False)
    
    
    def _generate_batch_data(self):
        while self.is_running:
            # Wait for training to start
            if not self.batch_generation_started.wait(timeout=0.1):
                continue
        
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            
            # If the number of samples is not a multiple of the batch size
            # then the last batch needs to wrap to the beginning of the sample indices
            wrap_length = (len(self) * self.batch_size) - self.n
            if self.shuffle:
                index_array = np.random.permutation(self.n)
                if wrap_length > 0:
                    index_array = np.concatenate((index_array, np.random.permutation(wrap_length))) 
            else:
                index_array = np.arange(self.n)
                if wrap_length > 0:
                    index_array = np.concatenate((index_array, np.arange(wrap_length))) 


            self.batch_data.start_batch()
    
            while self.batch_data.have_more_indices:
                while self.batch_data.request_count == 0 and self.batch_data.qsize() > self.max_batches_pending:
                    if self.is_shutdown:
                        return
                    if not self.batch_generation_started.is_set():
                        break
                    self.batch_data.wait()
                
                if not self.batch_generation_started.is_set():
                    break
                
                current_batch_index = self.batch_data.next_index()
                self.total_batches_seen += 1
                offset = current_batch_index*self.batch_size
                batch_index_chunk = index_array[offset:offset+self.batch_size]
                
                batch_filenames = []
                batch_classes = []
                
                for batch_index in batch_index_chunk:
                    batch_filenames.append(self.filenames[batch_index])
                    batch_classes.append(self.classes[batch_index])

                self._invoke_processing(current_batch_index, batch_filenames, batch_classes)


    def _invoke_processing(
        self, 
        batch_index:int, 
        batch_filenames:List[str], 
        batch_classes:List[int]
    ):
        try:
            self.pool(
                batch_index, 
                batch_filenames, 
                batch_classes, 
                params=self.process_params,
                pool_callback=self._pool_callback
            )
        except Exception as e:
            if not self.is_running:
                raise


    def _pool_callback(self, results):
        if results is not None:
            self.batch_data.put(results[0], results[1])
    


class ParallelProcessParams():
    """Adds methods related to getting batches from filenames

    It includes the logic to transform image files to batches.
    """

    def __init__(
        self,
        image_data_generator,
        target_size,
        batch_shape,
        color_mode,
        data_format,
        save_to_dir,
        save_prefix,
        save_format,
        subset,
        interpolation,
        class_indices,
        dtype,
        directory,
        class_mode,
        get_batch_function,
        preprocessing_function,
        noaug_preprocessing_function
    ):
        """Sets attributes to use later for processing files into a batch."""
        self.class_indices = class_indices
        self.dtype = dtype
        self.directory = directory
        self.class_mode = class_mode
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.batch_shape = batch_shape

        self.get_batch_function = get_batch_function
        self.preprocessing_function = preprocessing_function
        self.noaug_preprocessing_function = noaug_preprocessing_function

        
        if color_mode not in {'rgb', 'rgba', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb", "rgba", or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation
        self.validation_split = self.image_data_generator._validation_split
        if subset is not None:
            validation_split = self.image_data_generator._validation_split
            if subset == 'validation':
                split = (0, validation_split)
            elif subset == 'training':
                split = (validation_split, 1)
            else:
                raise ValueError(
                    'Invalid subset name: %s;'
                    'expected "training" or "validation"' % (subset,))
        else:
            split = None
        self.split = split
        self.subset = subset


    @property
    def n_classes(self) -> int:
        return len(self.class_indices)




def get_batches_of_transformed_samples(
    batch_index:int, 
    filenames:List[str], 
    classes:List[int], 
    params:ParallelProcessParams
) -> Tuple[int, Tuple[np.ndarray, np.ndarray]]:
    """Gets a batch of transformed samples.

    Arguments:
        batch_index: Index of this batch
        filenames: List of filenames for this batch
        classes: List of class ids mapping to the filenames list
        params: Generator parameters

    # Returns
        A batch of transformed samples: batch_index, (batch_x, batch_y)
    """

    # Ensure the RNG is unique for each batch
    if params.subset != 'validation' or params.image_data_generator.validation_augmentation_enabled:
        random.seed(batch_index + int(time.time()))
        np.random.seed(batch_index + int(time.time()))

    batch_size = params.batch_shape[0]
    assert len(filenames) == params.batch_shape[0]

    if isinstance(filenames[0], (list,tuple)):
        batch_x = []
        for _ in range(len(filenames[0])):
            batch_x.append(np.empty(params.batch_shape, dtype=params.dtype))
    else:
        batch_x = np.empty(params.batch_shape, dtype=params.dtype)


    # build batch of image data

    def _process_image_file(class_id, filename) -> np.ndarray:
        filepath = f'{params.directory}/{filename}'
        if filepath.endswith('npy'):
            x = np.load(filepath)

        else:
            # If set to 'None' don't rescale and let the user do it in preprocessing_func()
            if params.interpolation is None or params.interpolation.lower() == 'none':
                img = load_img(filepath,
                             color_mode=params.color_mode)
            else:
                img = load_img(filepath,
                             color_mode=params.color_mode,
                             target_size=params.target_size,
                             interpolation=params.interpolation)
            x = img_to_array(img, dtype='uint8')
        
            # Pillow images should be closed after `load_img`,
            # but not PIL images.
            if hasattr(img, 'close'):
                img.close()
        

        if params.noaug_preprocessing_function is not None:
            kwargs = _add_optional_callback_arguments( 
                params.noaug_preprocessing_function,
                batch_index=i,
                class_id=class_id,
                filename=filename,
                batch_class_ids=classes,
                batch_filenames=filenames
            )
            x = params.noaug_preprocessing_function(params, x, **kwargs)

        if params.subset != 'validation' or params.image_data_generator.validation_augmentation_enabled:
            transform_params = params.image_data_generator.get_random_transform(x.shape)
            x = x.astype(dtype=np.float32) # float required to do transform below
            x = params.image_data_generator.apply_transform(x, transform_params)
            
        else:
            x = img_to_array(x).astype(dtype=np.float32)
        if params.preprocessing_function is not None:
            kwargs = _add_optional_callback_arguments( 
                params.preprocessing_function,
                batch_index=i,
                class_id=class_id,
                filename=filename,
                batch_class_ids=classes,
                batch_filenames=filenames
            )
            x = params.preprocessing_function(params, x, **kwargs)

        x = params.image_data_generator.standardize(x)

        # optionally save augmented images to disk for debugging purposes
        if params.save_to_dir:
            img = array_to_img(x, params.data_format, scale=True)
            now = int(time.time() * 1000)
            fname = f'{filename.replace("/", "-")}_{now}.{params.save_format}'
            if params.save_prefix:
                fname = f'{params.save_prefix}_{fname}'
            img.save(os.path.join(params.save_to_dir, fname))            

        return x
           

    for i, filename in enumerate(filenames):
        if isinstance(filename, (list,tuple)):
            for j, fn in enumerate(filename):
                batch_x[j][i] = _process_image_file(i, fn)
        else:
            batch_x[i] = _process_image_file(i, filename)


    # build batch of labels
    if params.class_mode == 'input':
        batch_y = batch_x.copy()
    
    elif params.class_mode in {'binary', 'sparse'}:
        batch_y = np.empty((batch_size,), dtype=params.dtype)
        for i, clazz in enumerate(classes):
            batch_y[i] = clazz
    
    elif params.class_mode == 'categorical':
        batch_y = np.zeros((batch_size, params.n_classes), dtype=params.dtype)
        for i, clazz in enumerate(classes):
            batch_y[i, clazz] = 1.
        
    else:
        return batch_index, batch_x

    return batch_index, (batch_x, batch_y)






class BatchData:
    
    def __init__(
        self, 
        n:int, 
        shuffle:bool, 
        pool:ProcessPool
    ):
        self.n = n
        self.shuffle = shuffle
        self.pool = pool

        self.batch_data = queue.Queue() if shuffle else {}
        self.batch_data_lock = threading.Condition()
        self.indices_lock = threading.Condition()
        self.indices = []
        self.batch_counts = []
        self.saved_batch_zero = None
        self.requests = []
        self.data_event = threading.Event()
        
    
    @property
    def have_more_indices(self):
        with self.indices_lock:
            return (len(self.indices) + len(self.requests)) > 0
    
    @property
    def request_count(self):
        with self.indices_lock:
            return len(self.requests)
    
    
    def start_batch(self):
        with self.indices_lock:
            self.batch_counts.append(self.n)
            self.indices = [i for i in range(self.n)]
    
    
    def next_index(self):
        with self.indices_lock:
            if len(self.requests) > 0:
                idx = self.requests.pop(0)
                try:
                    self.indices.remove(idx)
                except:
                    pass 
                
                return idx
            
            else:
                return self.indices.pop(0)
        
    
    def wait(self):
        self.data_event.clear()
        while self.pool.is_running:
            if self.data_event.wait(timeout=.1):
                return True 
        return False
        
    
    def reset(self):
        if self.shuffle:
            while not self.batch_data.empty():
                self.batch_data.get()
        else:
            with self.batch_data_lock:
                self.batch_data.clear()
                
        with self.indices_lock:
            self.requests = []
            self.indices = [i for i in range(self.n)]
        
    
    def qsize(self):
        if self.shuffle:
            return self.batch_data.qsize()
        else:
            with self.batch_data_lock:
                return len(self.batch_data)

     
    def put(self, index, value):
        if self.shuffle:
            self.batch_data.put(value)
        else:
            with self.batch_data_lock:
                self.batch_data[index] = value
                self.batch_data_lock.notify_all()
    
     
    def get(self, index, save_batch_zero=False):
        decrement_batch_count = True 

        # If we're returning batch zero and we have a saved one,
        # then just return that batch
        if index == 0 and self.saved_batch_zero is not None:
            retval = self.saved_batch_zero
            self.saved_batch_zero = None

        elif self.shuffle:
            while True:
                if not self.pool.is_running:
                    raise StopIteration('The data generator has been stopped')

                try:
                    retval = self.batch_data.get(timeout=0.1)
                    break
                except queue.Empty:
                    continue

        else:
            with self.batch_data_lock:
                if index not in self.batch_data:
                    with self.indices_lock:
                        self.requests.append(index)
                        self.data_event.set()
                        
                    while index not in self.batch_data:
                        if not self.pool.is_running:
                            raise StopIteration('The data generator has been stopped')
                        self.batch_data_lock.wait(timeout=0.1)
                    
                retval = self.batch_data[index]
                del self.batch_data[index]


        # If this is batch0 and we should save it
        # then saved a reference to it and do NOt decrement the batch count as it will be returned in a later call
        if index == 0 and save_batch_zero:
            self.saved_batch_zero = retval
            decrement_batch_count = False

        is_last_in_batch = self._decrement_current_batch_count() if decrement_batch_count else False
        self.data_event.set()

        return retval, is_last_in_batch


    def _decrement_current_batch_count(self) -> bool:
        is_last_in_batch = False
        with self.indices_lock:
            current_count = self.batch_counts[0]
            if current_count == 1:
                is_last_in_batch = True 
                self.batch_counts.pop(0)
            else:
                self.batch_counts[0] = current_count - 1
            
        return is_last_in_batch



def _add_optional_callback_arguments(
    func, 
    batch_index, 
    class_id, 
    filename, 
    batch_class_ids, 
    batch_filenames
) -> dict:
    retval = {}
    args = inspect.getfullargspec(func).args
    if 'batch_index' in args:
        retval['batch_index'] = batch_index
    if 'class_id' in args:
        retval['class_id'] = class_id
    if 'filename' in args:
        retval['filename'] = filename
    if 'batch_class_ids' in args:
        retval['batch_class_ids'] = batch_class_ids
    if 'batch_filenames' in args:
        retval['batch_filenames'] = batch_filenames

    return retval

