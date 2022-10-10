import sys
import logging
import signal
import traceback
import functools
import os
import multiprocessing
import warnings
from multiprocessing import Pool 
from multiprocessing.pool import ThreadPool
import threading


from .logger import get_logger




class ProcessPoolManager(object):
    """This allows for running Python functions across multiple, independent processes"""
    warnings.warn("The mltk.utils.ProcessPoolManager class has been deprecated. See mltk.utils.process_pool.ProcessPool for a more optimal implementation", DeprecationWarning)

    @staticmethod
    def set_logger(logger: logging.Logger):
        globals()['_logger'] = logger

   
    def __init__(
        self, 
        callback=None, 
        cores=-1, 
        pool: Pool=None, 
        debug=False,
        logger:logging.Logger=None,
        env:dict=None
    ):
        max_cores = multiprocessing.cpu_count()

        logger = logger or globals()['_logger'] or get_logger()

        if cores == -1:
            cores = max_cores
        elif isinstance(cores, float):
            cores = round(max_cores * cores)
        
        cores = min(max(cores, 1), max_cores)
        
        if pool is None:   
            logger.info(
                f'ProcessPoolManager using {cores} of {max_cores} CPU cores\n'
                'NOTE: You may need to adjust the "cores" parameter of the data generator if you\'re experiencing performance issues'
            )
            # ThreadPool is easier to debug as it allows for single-stepping
            if debug:
                logger.debug('NOTE: ProcessPoolManager using ThreadPool (instead of ProcessPool)')
                pool = ThreadPool(processes=cores)  
            else:
                pool = Pool(processes=cores, initializer=functools.partial(_init_pool_worker, env))
            
        self.logger = logger
        self.pool = pool
        self.callback = callback
        self._pending_count = 0
        self._max_pending_count = cores + 2
        self._pending_lock = threading.Condition()
        self._reset_active = threading.Event()
        self._is_closed = threading.Event()
        self._consecutive_errors = 0
        

    def reset(self):
        """Reset all processes and clear any pending"""
        self._reset_active.set()
        with self._pending_lock:
            self._pending_lock.notify_all()
        self.wait(timeout=3)
        self._reset_active.clear()

        
    def wait(self, timeout=None) -> bool:
        """Wait for all processes to complete"""
        if self._is_closed.is_set():
            return False

        with self._pending_lock:
            while self._pending_count > 0:
                if not self._pending_lock.wait(timeout=timeout):
                    return False 
        return True
        
        
    def close(self):
        """Close the processing pool"""
        self._is_closed.set()
        self.pool.close()
        self.pool.terminate()
        

    def process(self, func, *args, **kwargs):
        """Process the given function in the process pool"""
        with self._pending_lock:
            if  self._consecutive_errors > 1:
                raise RuntimeError('Max subprocess consecutive errors exceeded. Aborting.')

            if self._reset_active.is_set() or self._is_closed.is_set():
                return 
            
            while self._pending_count >= self._max_pending_count:
                if self._reset_active.is_set():
                    return
                self._pending_lock.wait(timeout=.1)   
            self._pending_count += 1
        
        kwargs['__func__'] = func

        if self._is_closed.is_set():
            return 
        
        try:
            self.pool.apply_async(
                _on_process, args, kwargs, 
                callback=self._on_complete, 
                error_callback=self._on_error
            )
        except Exception as e: 
            if 'Pool not running' in f'{e}':
                return 
            raise e 


    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.wait()
        self.close()


    def _on_complete(self, result):
        try:
            
            if self.callback is not None and not self._reset_active.is_set():
                try:
                    self.callback(result)
                except KeyboardInterrupt:
                    sys.exit()
                except Exception as e:
                    self.logger.error(f'Error in pool callback {self.callback}, err: {e}', exc_info=e)
            
        finally:
            with self._pending_lock:
                self._consecutive_errors = 0
                self._pending_count -= 1
                self._pending_lock.notify_all()

    
    
    def _on_error(self, e):
        with self._pending_lock:
            self._consecutive_errors += 1
            self._pending_count -= 1
            self._pending_lock.notify_all()
            if  self._consecutive_errors == 1:
                self.logger.error(
                    'Error in pool subprocess\n\n'
                    'HINT: Use the debug=True option to single-step debug the following stacktrace\n\n'
                    f'Error details: {e}'
                )


def _on_process(*args, **kwargs):
    """Process the given function in the current subprocess"""
    try:
        func = kwargs['__func__']
        del kwargs['__func__']
        return func(*args, **kwargs)
    except KeyboardInterrupt:
        sys.exit()
    except Exception as e:
        raise type(e)(traceback.format_exc())

# Set up the worker processes to ignore SIGINT altogether, 
# and confine all the cleanup code to the parent process. 
# This fixes the problem for both idle and busy worker processes, 
# and requires no error handling code in your child processes.
def _init_pool_worker(env:dict):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if env:
        os.environ.update(env)



if '_logger' not in globals():
    _logger = None
