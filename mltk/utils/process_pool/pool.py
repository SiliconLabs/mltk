from __future__ import annotations
from typing import Callable, List, Union,  Dict
import sys
import os
import atexit
import threading
import queue
import subprocess
import copy
import logging
from  multiprocessing import cpu_count as get_cpu_count

from ._utils import (read_data, write_data)


class ProcessPool:
    """Parallel Processing Pool

    This allows for executing a Python function across multiple CPU cores.
    The function executes in separate processes and thus is not limited by
    the Global Interpreter Lock (GIL) which can slow down parallel processing
    when using Python threads.

    There are three ways of using this utility.

    .. highlight:: python
    .. code-block:: python

        # Assume we have the processing function at the root of a Python module:
        def my_processing_func(x):
            y = x * x
            return y

        # And we instantiate the pool as
        pool = ProcessPool(my_processing_func, 0.5)


        # ----------------------------
        # Callback Driven
        #
        # Add all of the processing data to the pool and invoke a callback for each result

        results = []
        def _processing_callback(y):
            results.append(y)

        # Process each entry in a subprocess
        # and call the callback when the results is available
        for i in range(1000):
            pool(i, pool_callback=_processing_callback)

        # Main thread waits for the results to complete
        while len(results) != 1000:
            time.sleep(1)


        # ----------------------------
        # Batch Driven
        #
        # Process the data in batches and wait for all of the results

        # Create a batch
        batch = pool.create_batch(1000)

        # Process each entry in a subprocess
        # Internally the given batch will be populated with the results
        for i in range(1000):
            pool(i, pool_batch=batch)

        # Main thread waits for the results to complete
        results = batch.wait()

        # ----------------------------
        # Wait for each result
        #
        # Process the data and wait for the result
        # NOTE: In this mode, the pool should be invoked from separate threads to see a speedup
        # from the parallel processing

        results = []
        for i in range(1000):
            y = pool(i)
            results.append(y)


    See the source code on Github: `mltk/utils/process_pool <https://github.com/siliconlabs/mltk/blob/master/mltk/utils/process_pool>`_

    Args:
        entry_point: The Python function to execute in separate subprocesses.
            This function must exist at the root of its module (i.e. it must not be nested or a class method)
            Typically the function will accept one or more arguments, process the data, and return a result
        n_jobs: The number of subprocesses to use for processing. Typically more jobs means faster processing at the expends of more RAM/CPU usage
        name: The name to prefix to the python threads used to monitor the subprocesses
        start: Automatically start the subprocesses. If false, then the start() must be called before processing
        debug: If true then the entry_point is executed in a single thread instead of multiple subprocesses. This reduces complexity which can aid debugging
        env: The OS environment variables to export in the subprocesses
        disable_gpu_in_subprocesses: Disables NVidia GPU usage in the subprocesses. This is necessary if the Tensorflow python package is imported in the entry_point's module
        logger: Optional Python logger
    """
    def __init__(
        self,
        entry_point:Callable,
        n_jobs:int,
        name='ProcessPool',
        start=True,
        debug=False,
        env:Dict[str,str]=None,
        disable_gpu_in_subprocesses=True,
        logger:logging.Logger=None
    ):
        if os.environ.get('MLTK_PROCESS_POOL_SUBPROCESS', ''):
            return

        self.logger = logger or logging.getLogger(name)

        self._n_jobs = 1 if debug else calculate_n_jobs(n_jobs)
        self._name = name
        self._entry_point = entry_point
        self._running_event = threading.Event()
        self._ready_q = queue.Queue(maxsize=self._n_jobs)
        self._processes:List[_Subprocess] = []
        self._debug = debug
        self._env = copy.deepcopy(env) if env else {}
        self._lock = threading.Lock()
        self._detected_pthread_error = False
        self._detected_subprocess_error:str = None

        self.logger.info(f'{self.name} is using {self.n_jobs} subprocesses')

        if disable_gpu_in_subprocesses:
            self._env['CUDA_VISIBLE_DEVICES'] = '-1'
            self._env['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Limit the number of thread spun up in the subprocesses
        # as they can quickly deplete the system resources.
        # NOTE: If any of these variables are defined in the env arg or global env
        #       then that value will be used instead
        MAX_NUM_THREADS_VARS = [
            'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS',
            'NUMEXPR_NUM_THREADS',
        ]
        for key in MAX_NUM_THREADS_VARS:
            self._env[key] = self._env.get(key, os.environ.get(key, '1'))


        atexit.register(self.shutdown)

        if start:
            self.start()

    @property
    def n_jobs(self) -> int:
        """The number of subprocesses used"""
        return self._n_jobs

    @property
    def name(self) -> str:
        """The name of this processing pool instance"""
        return self._name

    @property
    def is_running(self) -> bool:
        """Returns true if the processing pool is actively running"""
        return self._running_event.is_set()

    @property
    def detected_subprocess_error(self) -> str:
        with self._lock:
            return self._detected_subprocess_error
    @detected_subprocess_error.setter
    def detected_subprocess_error(self, v:str):
        with self._lock:
            if self._detected_subprocess_error is None:
                self._detected_subprocess_error = v


    def start(self):
        """Start the processing pool subprocesses"""
        if self.is_running:
            raise RuntimeError('Process pool already running')

        self._running_event.set()

        for i in range(self._n_jobs):
            subprocess = _Subprocess(
                name=f'{self._name}-ProcessPool-{i}',
                pool=self,
                entry_point=self._entry_point,
                debug=self._debug,
                env=self._env,
                logger=self.logger
            )
            self._processes.append(subprocess)
            self._ready_q.put(subprocess)


    def shutdown(self):
        """Shutdown the processing pool subprocesses immediately"""
        if self.is_running:
            self._running_event.clear()
            for subprocess in self._processes:
                subprocess.shutdown()

            if self._detected_pthread_error:
                self.logger.warning(
                    '\n***\nYour system may be running low on resources.\n'
                    'Trying reducing the n_jobs (or cores) used for parallel processing.\n***\n\n'
                )

    def __enter__(self):
        if not self.is_running:
            self.start()
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


    def __call__(
        self,
        *args,
        pool_callback:Callable=None,
        pool_batch:ProcessPoolBatch=None,
        **kwargs
    ):
        try:
           return self.process(
            *args,
            pool_callback=pool_callback,
            pool_batch=pool_batch,
            **kwargs
        )
        except:
            self.shutdown()
            raise

    def create_batch(self, size:int) -> ProcessPoolBatch:
        """Create a processing pool batch.
        See this class's docs for more info
        """
        return ProcessPoolBatch(pool=self, size=size)


    def process(
        self,
        *args,
        pool_callback:Callable=None,
        pool_batch:ProcessPoolBatch=None,
        **kwargs
    ) -> Union[ProcessPoolBatch,object]:
        """Process the given args, kwargs in the next available subprocess
        See this class's docs for more info
        """
        if not self.is_running:
            raise RuntimeError('ProcessPool not started')

        batch = pool_batch or ProcessPoolBatch(self, pool_callback=pool_callback)

        while True:
            if not self.is_running:
                raise RuntimeError('ProcessPool shutdown')

            try:
                subprocess:_Subprocess = self._ready_q.get(block=True, timeout=0.100)
                break
            except queue.Empty:
                continue

        subprocess.invoke(args, kwargs, batch=batch)

        if pool_callback is None and pool_batch is None:
            results = batch.wait()
            if not self.is_running:
                raise RuntimeError('ProcessPool shutdown')

            return results

        return batch




class ProcessPoolBatch:
    """"Processing Pool Batch
    This is used to store the results of a processing batch of data.
    """
    def __init__(
        self,
        pool:ProcessPool,
        pool_callback:Callable=None,
        size:int=0
    ):
        self.pool = pool
        self.pool_callback = pool_callback
        self._condition = threading.Condition()
        self._size = size
        self._return_scalar = size == 0
        self._remaining = max(1, size)
        self._next_index_count = 0
        self._results = [None for _ in range(self._remaining)]

    @property
    def size(self) -> int:
        """The size of the batch"""
        return self._size

    def wait(self) -> Union[object,List[object]]:
        """Wait for all of the data in the processing batch to complete and return the results.

        Returns:
            Either a single object if size=0 or a list of objects if size>0
        """
        with self._condition:
            while self._remaining > 0:
                if not self.pool.is_running:
                    return None
                self._condition.wait()

            return self._results[0] if self._return_scalar else self._results


    def _next_index(self) -> int:
        with self._condition:
            if self._remaining <= 0:
                raise RuntimeError('Batch overflow')

            retval = self._next_index_count
            self._next_index_count += 1

        return retval

    def _add_results(self, index:int, results):
        if self.pool_callback is not None:
            if index != -1:
                self.pool_callback(results)
            return

        with self._condition:
            if index != -1:
                self._results[index] = results
                self._remaining -= 1
            self._condition.notify()






class _Subprocess(threading.Thread):
    def __init__(
        self,
        name:str,
        pool:ProcessPool,
        entry_point:Callable,
        debug:bool,
        env:Dict[str,str],
        logger:logging
    ):
        threading.Thread.__init__(
            self,
            name=name,
            daemon=True,
            target=self._process_thread_loop
        )

        self.pool = pool
        self.logger = logger
        self._entry_point = entry_point
        self._invoke_sem = threading.Semaphore(value=0)
        self._invoke_args=None
        self._invoke_kwargs=None
        self._invoke_batch:ProcessPoolBatch=None
        self._invoke_batch_index:int=-1
        self._shutdown_event = threading.Event()

        if not debug:
            curdir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
            subprocess_main_path = f'{curdir}/_subprocess_main.py'
            os_env = os.environ.copy()
            os_env.update(dict(
                MODULE_PATH=entry_point.__code__.co_filename,
                FUNCTION_NAME=entry_point.__name__,
                PROCESS_POOL_NAME=name,
            ))
            os_env.update(env)
            self._subprocess = subprocess.Popen(
                [sys.executable, '-u', subprocess_main_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                env=os_env,
                close_fds=True,
                bufsize=-1,
            )
            self._monitor_thread = threading.Thread(
                name=f'{name}-monitor',
                target=self._monitor_process_logs,
                daemon=True
            )
            self._monitor_thread.start()

        else:
            self._subprocess = None
            self._process_thread_loop_unsafe = self._debug_thread_loop_unsafe

        self.start()


    def invoke(
        self,
        args, kwargs,
        batch:ProcessPoolBatch
    ):
        assert self._invoke_args is None, 'Bad processing Q state'

        self._invoke_args = args
        self._invoke_kwargs = kwargs
        self._invoke_batch = batch
        self._invoke_batch_index = batch._next_index()
        self._invoke_sem.release()


    def _process_thread_loop(self):
        try:
            self._process_thread_loop_unsafe()
        except KeyboardInterrupt:
            pass
        except OSError:
            pass
        except Exception as e:
            if self.pool.is_running:
                self.logger.error(f'{self.name}: {e}', exc_info=e)
        finally:
            self.pool.shutdown()


    def _process_thread_loop_unsafe(self):
        while True:
            self._invoke_sem.acquire()

            retcode = self._subprocess.poll()
            if retcode is not None:
                if retcode != 0:
                    raise RuntimeError(f'{self.name} terminated with error code: {retcode}')
                return

            write_data(
                self._subprocess.stdin,
                self._invoke_args,
                self._invoke_kwargs,
            )
            self._invoke_args = None
            self._invoke_kwargs = None

            result, _ = read_data(
                self._subprocess.stdout
            )

            # If the subprocess failed to return data
            if len(result) == 0:
                # Wait a moment for the subprocess to complete
                self._subprocess.wait(1)
                # Poll the subprocesses exit code
                retcode = self._subprocess.poll()

                # If an exit code was returned
                if retcode is not None:
                    # If the retcode is non-zero then the subprocess failed
                    # So throw an exception
                    if retcode != 0:
                        raise RuntimeError(f'{self.name} terminated with error code: {retcode}')

                    # Otherwise just return as the subprocess is being gracefully terminated
                    return

                # Otherwise the subprocess is not properly generating data
                # So throw an exception
                raise RuntimeError(f'{self.name} did not return a result')

            if isinstance(result, (tuple,list)) and len(result) == 1:
                result = result[0]

            self._invoke_batch._add_results(self._invoke_batch_index, result)
            self._invoke_batch = None
            self._invoke_batch_index = -1
            self.pool._ready_q.put(self)


    def _debug_thread_loop_unsafe(self):
        while True:
            self._invoke_sem.acquire()
            if self._shutdown_event.is_set():
                break

            result = self._entry_point(*self._invoke_args, **self._invoke_kwargs)
            result = result or (None, )
            self._invoke_args = None
            self._invoke_kwargs = None

            if len(result) == 0:
                raise RuntimeError(f'{self.name} did not return a result')

            if isinstance(result, (tuple,list)) and len(result) == 1:
                result = result[0]

            self._invoke_batch._add_results(self._invoke_batch_index, result)
            self._invoke_batch = None
            self._invoke_batch_index = -1
            self.pool._ready_q.put(self)



    def shutdown(self):
        try:
            # First try to gracefully terminate the subprocess (e.g. issue Ctrl+C)
            # and wait a moment for it to complete
            self._subprocess.terminate()
            self._subprocess.wait(1)
        except:
            pass
        try:
            # Close the data input pipe
            self._subprocess.stdin.close()
        except:
            pass
        try:
            # Close the data output pipe
            self._subprocess.stdout.close()
        except:
            pass
        try:
            # Wait a moment for the log monitor thread to complete
            self._monitor_thread.join(3)
        except:
            pass
        try:
            # Close the log pipe
            self._subprocess.stderr.close()
        except:
            pass
        try:
            # Forcefully kill the subprocess if necessary
            self._subprocess.kill()
        except:
            pass

        self._shutdown_event.set()
        self._invoke_sem.release()
        batch = self._invoke_batch
        self._invoke_batch = None
        if batch is not None:
            batch._add_results(-1, None)


    def _monitor_process_logs(self):
        try:
            while True:
                line = self._subprocess.stderr.readline()
                if not line:
                    return
                if not self.pool.detected_subprocess_error and self._subprocess.poll():
                    self.pool.detected_subprocess_error = self.name

                detected_subprocess_error = self.pool.detected_subprocess_error
                if (self.pool.is_running and detected_subprocess_error is None) or detected_subprocess_error == self.name:
                    line = line.decode('utf-8').rstrip()
                    if not self.pool._detected_pthread_error and 'pthread_create() failed' in line:
                        self.pool._detected_pthread_error = True
                    if 'Traceback' in line:
                        self.pool.detected_subprocess_error = self.name
                    self.logger.info(f'{self.name}: {line}')
        except:
            pass





def calculate_n_jobs(n_jobs:Union[float,int]) -> int:
    """Calculate the number of subprocesses to use for the processing pool

    Args:
        n_jobs: This should be one of:
            - A float value between (0, 1.0], which specifies the percentage of the available CPUs
            - Integer specifying the exact number of CPUs. This will automatically clamp to the maximum CPUs in the system if necessary

    Returns:
        The calculated number of jobs to use for subprocessing
    """
    n_cpus = get_cpu_count()

    if n_jobs == -1:
        n_jobs = n_cpus
    elif isinstance(n_jobs, float):
        if n_jobs < 0 or n_jobs > 1:
            raise ValueError('Must either be an integer or a float in the range (0, 1.0]')
        n_jobs = round(n_cpus * n_jobs)

    return min(max(n_jobs, 1), n_cpus)


