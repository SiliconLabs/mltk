"""Utilities for real-time data augmentation on image data.
"""
import os
import time
import threading
import inspect
import queue
from typing import List, Tuple
import numpy as np
import random



from keras_preprocessing.image.utils import (
    array_to_img,
    img_to_array,
    load_img
)
from mltk.core import get_mltk_logger
from mltk.core.keras import DataSequence
from mltk.utils.process_pool_manager import ProcessPoolManager


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
        self.batch_generation_started = threading.Event()
        self.batch_generation_shutdown = threading.Event()

        self.batch_data = BatchData(len(self), shuffle, shutdown_event=self.batch_generation_shutdown)
       
        t = threading.Thread(
            target=self._generate_batch_data_safe, 
            name=f'Batch data generator:{process_params.subset}'
        )
        t.setDaemon(True)
        t.start()

        pool_manager_kwargs = dict(
            cores=self.cores, 
            callback=self._on_batch_data_ready,
            debug=self.debug,
        )

        # If specified,
        # Add the CUDA_VISIBLE_DEVICES=-1 environment variable
        # so TF doesn't use the GPU in the subprocesses
        if self.disable_gpu_in_subprocesses:
            pool_manager_kwargs['env'] = dict(
                CUDA_VISIBLE_DEVICES='-1',
            )

        # Re-use the processing pool if one has already been created
        if '_multiprocess_pool' in globals():
            pool_manager_kwargs['pool'] = globals()['_multiprocess_pool']

        self.pool = ProcessPoolManager(**pool_manager_kwargs) 
        globals()['_multiprocess_pool'] = self.pool.pool 
        

    @property
    def batch_size(self) -> int:
        return self.batch_shape[0]
    
        
    def reset(self):
        self.batch_generation_started.clear()
        self.batch_data.reset()
        self.pool.reset()
        self.batch_data.reset()
        self.batch_index = 0
    
    
    def shutdown(self):
        self.batch_generation_shutdown.set()
        self.reset()
        self.pool.close()
        if '_multiprocess_pool' in globals():
            del globals()['_multiprocess_pool']


    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))

        if self.batch_generation_shutdown.is_set():
            raise Exception('Data generator has been shutdown')

        self.batch_generation_started.set()
        
        return self.batch_data.get(idx)


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
            self.batch_generation_started.clear()
            raise StopIteration()
        
        if self.batch_generation_shutdown.is_set():
            raise Exception('Data generator has been shutdown')
        
        self.batch_generation_started.set()
        retval = self.batch_data.get(self.batch_index)
        self.batch_index += 1
        
        return retval


    def _generate_batch_data_safe(self):
        try:
            self._generate_batch_data()
        except Exception as e:
            get_mltk_logger().error(f'Exception during batch data processing, err: {e}', exc_info=e)
            self.shutdown()
    
    
    def _generate_batch_data(self):
        while not self.batch_generation_shutdown.is_set():
            # Wait for training to start
            if not self.batch_generation_started.wait(timeout=0.1):
                continue
        
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            
            if self.shuffle:
                index_array = np.random.permutation(self.n)
            else:
                index_array = np.arange(self.n)
            
    
            while self.batch_data.have_more_indices:
                while self.batch_data.request_count == 0 and self.batch_data.qsize() > self.max_batches_pending:
                    self.batch_data.wait()
                
                if not self.batch_generation_started.is_set():
                    break
                
                
                idx = self.batch_data.next_index()
                self.total_batches_seen += 1
                offset = idx*self.batch_size
                batch_index_chunk = index_array[offset:offset+self.batch_size]
                
                batch_filenames = []
                batch_classes = []
                
                for batch_index in batch_index_chunk:
                    batch_filenames.append(self.filenames[batch_index])
                    batch_classes.append(self.classes[batch_index])
                
                if not self.batch_generation_shutdown.is_set():
                    get_batch_function = self.process_params.get_batch_function or get_batches_of_transformed_samples
                    self.pool.process(
                        get_batch_function, 
                        idx,
                        batch_filenames, 
                        batch_classes, 
                        self.process_params
                    )
            
            
            self.batch_data.reset_indices()


    def _on_batch_data_ready(self, result):
        self.batch_data.put(result[0], result[1])





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
        """Sets attributes to use later for processing files into a batch.

        # Arguments
            image_data_generator: Instance of `ImageDataGenerator`
                to use for random transformations and normalization.
            target_size: tuple of integers, dimensions to resize input images to.
            color_mode: One of `"rgb"`, `"rgba"`, `"grayscale"`.
                Color mode to read images.
            data_format: String, one of `channels_first`, `channels_last`.
            save_to_dir: Optional directory where to save the pictures
                being yielded, in a viewable format. This is useful
                for visualizing the random transformations being
                applied, for debugging purposes.
            save_prefix: String prefix to use for saving sample
                images (if `save_to_dir` is set).
            save_format: Format to use for saving sample images
                (if `save_to_dir` is set).
            subset: Subset of data (`"training"` or `"validation"`) if
                validation_split is set in ImageDataGenerator.
            interpolation: Interpolation method used to resample the image if the
                target size is different from that of the loaded image.
                Supported methods are "nearest", "bilinear", and "bicubic".
                If PIL version 1.1.3 or newer is installed, "lanczos" is also
                supported. If PIL version 3.4.0 or newer is installed, "box" and
                "hamming" are also supported. By default, "nearest" is used.
        """
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
    random.seed(batch_index)
    np.random.seed(batch_index)

    batch_size = params.batch_shape[0]

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
            x = x.astype(dtype='float32') # float required to do transform below
            x = params.image_data_generator.apply_transform(x, transform_params)
            
        else:
            x = img_to_array(x)

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






class BatchData(object):
    
    def __init__(self, n, shuffle, shutdown_event: threading.Event):
        self.n = n
        self.shuffle = shuffle
        self.batch_data = queue.Queue() if shuffle else {}
        self.batch_data_lock = threading.Condition()
        self.indices_lock = threading.Condition()
        self.indices = [i for i in range(self.n)]
        self.requests = []
        self.data_event = threading.Event()
        self.shutdown_event = shutdown_event
    
    
    @property
    def have_more_indices(self):
        with self.indices_lock:
            return (len(self.indices) + len(self.requests)) > 0
    
    
    @property
    def request_count(self):
        with self.indices_lock:
            return len(self.requests)
    
    
    def reset_indices(self):
        with self.indices_lock:
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
        while not self.shutdown_event.is_set():
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
    
     
    def get(self, index):
        if self.shuffle:
            while True:
                if self.shutdown_event.is_set():
                    return None
                try:
                    retval = self.batch_data.get(timeout=0.1)
                    break
                except queue.Empty:
                    continue
        
        else:
            with self.batch_data_lock:
                if not index in self.batch_data:
                    with self.indices_lock:
                        self.requests.append(index)
                        self.data_event.set()
                        
                    while index not in self.batch_data:
                        if self.shutdown_event.is_set():
                            return None 
                        self.batch_data_lock.wait(timeout=0.1)
                    
                retval = self.batch_data[index]
                del self.batch_data[index]

        self.data_event.set()

        return retval
    

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