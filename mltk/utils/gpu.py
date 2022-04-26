import os
import sys
import re
import atexit
import logging
from collections import namedtuple

from .logger import DummyLogger
from .python import SHORT_VERSION


# See:
# https://www.tensorflow.org/install/source#gpu
TensorflowCudaVersions = namedtuple('TensorflowCudaVersions', ['tensorflow', 'cudnn', 'cuda', 'python_min', 'python_max'])
TENSORFLOW_CUDA_COMPATIBILITY = [
    # This should be in DESCENDING order of the Tensorflow version
    TensorflowCudaVersions('2.8', '8.1', '11.2', '3.7', '3.10'),
    TensorflowCudaVersions('2.7', '8.1', '11.2', '3.7', '3.9'),
    TensorflowCudaVersions('2.6', '8.1', '11.2', '3.7', '3.9'),
    TensorflowCudaVersions('2.5', '8.1', '11.2', '3.7', '3.9'),
    TensorflowCudaVersions('2.4', '8.0', '11.0', '3.7', '3.8'),
    TensorflowCudaVersions('2.3', '7.6', '10.1', '3.7', '3.8'),
]

if '_selected_gpu_id' not in globals():
    _selected_gpu_id = -1 


def disable():
    """Disable the GPU from being used by Tensorflow"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def initialize(logger=None):
    """Initialize the GPU for usage with Tensorflow

    NOTE: The deinitialize() API will automatically be called when the script exits
    """
    _selected_gpu_id = globals()['_selected_gpu_id']
    CUDA_VISIBLE_DEVICES = os.getenv('CUDA_VISIBLE_DEVICES', None)
    if CUDA_VISIBLE_DEVICES == '-1':
        return

    if _selected_gpu_id != -1:
        return 

    logger = logger or DummyLogger()


    try:
        # %% Select available GPU
        import GPUtil
        import tensorflow as tf 

        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        gpus = GPUtil.getGPUs()
        
        if len(gpus) == 0:
            logger.info("No GPUs found, using CPU for training")
            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
            return
        
        max_gpu = gpus[0]
        for gpu in gpus[1:]:
            if gpu.memoryFree > max_gpu.memoryFree:
                max_gpu = gpu
        
        os.environ['CUDA_VISIBLE_DEVICES'] = str(max_gpu.id)
        globals()['_selected_gpu_id'] = max_gpu.id

        # Enable dynamic memory growth for GPU
        try:
            tf_gpus = tf.config.list_physical_devices('GPU')
            if len(tf_gpus) == 0:
                logger.warning('\n\n\n'
                    '*******************************************************************************\n'
                    'WARNING: Failed to load GPU driver\n'
                    '\n'
                    'This could mean that the driver or CUDA libraries are not properly installed.\n'
                    'Refer to the Tensorflow GPU installation guide here:\n'
                    'https://www.tensorflow.org/install/gpu\n'
                    '\n'
                    'Alternatively, you can disable the GPU by defining the environment variable: CUDA_VISIBLE_DEVICES=-1\n'
                    '.e.g.:\n'
                    f'{"set" if os.name == "nt" else "export"} CUDA_VISIBLE_DEVICES=-1\n\n'
                    '*******************************************************************************\n'
                )
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                deinitialize(force=True)
                return 

            tf.config.experimental.set_memory_growth(tf_gpus[max_gpu.id], True)
        except Exception as e:
            logger.debug(f'tf.config.experimental.set_memory_growth() failed, err: {e}')

        # The TfLiteConverter adds a StreamHandler to the root logger, 
        # remove it so we don't double print everything to the console
        logging.getLogger().handlers.clear()

        logger.debug(f"Selected GPU : {max_gpu.name} (id={max_gpu.id})")
        atexit.register(deinitialize)

    except Exception as e:
        logger.warning(f'GPU init err: {e}')
        logger.info("Using CPU for training")
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
        deinitialize()


def deinitialize(force=False):
    """De-initialize the GPU

    NOTE: This is automatically called when the script exits
    """
    _selected_gpu_id = -1 if '_selected_gpu_id' not in globals() else globals()['_selected_gpu_id']

    if _selected_gpu_id != -1:
        try:
            from numba import cuda

            cuda.select_device(_selected_gpu_id)
            cuda.close()
        except:
            pass

        globals()['_selected_gpu_id'] = -1





def get_tensorflow_version_with_cudnn_version(cudnn_ver:str) -> TensorflowCudaVersions:
    toks = cudnn_ver.split('.')
    cudnn_ver = '.'.join(toks[:2]) # strip of the "patch" part of the version string

    # Find the largest Tensorflow version that it
    # compatible with the given CuDNN version
    for ver in TENSORFLOW_CUDA_COMPATIBILITY:
        if ver.cudnn == cudnn_ver:
            return ver

    return None


def check_tensorflow_cuda_compatibility_error(log_file_path:str) -> str:
    try:
        with open(log_file_path, 'r') as f:
            log_lines = f.read().splitlines()
    except:
        return None 

    required_tensorflow_version = None 
    invalid_gpu_driver = False
    cuda_error_re = re.compile(r'.*Loaded runtime CuDNN library: (\d+\.\d+\.\d+) but source was compiled with: (\d+\.\d+\.\d+).*')
    for line in log_lines:
        match = cuda_error_re.match(line)
        if match:
            installed_cudnn_ver = match.group(1)
            expected_cudnn_ver = match.group(2)
            required_tensorflow_version = get_tensorflow_version_with_cudnn_version(installed_cudnn_ver)
        elif 'DNN library is not found' in line:
            invalid_gpu_driver = True 


    if not(required_tensorflow_version or invalid_gpu_driver):
        return None 

    def _current_python_version_supported(ver:TensorflowCudaVersions) -> bool:
        def _version_to_int(v):
            toks = v.split('.')
            return int(toks[0]) * 100 + int(toks[1])

        current_python = _version_to_int(SHORT_VERSION)
        python_min = _version_to_int(ver.python_min)
        python_max = _version_to_int(ver.python_max)

        return current_python >= python_min and current_python <= python_max


    retval  = 'There appears to be a compatibility issue with MLTK Python venv Tensorflow version and installed GPU driver.\n'
    retval += 'For a compatibility list, see:\n'
    retval += 'https://www.tensorflow.org/install/source#gpu\n\n'
    retval += 'Recommended solutions:\n\n'
    count = 1
    if required_tensorflow_version is not None:
        retval += f'{count}. Update the Tensorflow version in the MLTK Python venv to match the installed GPU driver by running:\n'
        retval += f'   {"pip" if os.name == "nt" else "pip3"} install tensorflow=={required_tensorflow_version.tensorflow}.*\n'

        if not _current_python_version_supported(required_tensorflow_version):
            retval += '\n'
            retval += f'   NOTE: Your current Python version: {SHORT_VERSION} is NOT supported by Tensorflow-{required_tensorflow_version.tensorflow}\n'
            retval +=  '   To resolve this issue:\n'
            retval += f'   1. Created a new Python virtual environment using a Python version {required_tensorflow_version.python_min}-{required_tensorflow_version.python_max}\n'
            retval +=  '   2. Install the MLTK Python package\n'
            retval += f'   3. Install Tensorflow-{required_tensorflow_version.tensorflow}\n'

        retval += '\n'

        count += 1 

    retval += f'{count}. Update your GPU driver to match the installed Tensorflow version in the MLTK venv, see:\n'
    retval += '   https://www.tensorflow.org/install/gpu\n'
    retval += '   https://www.tensorflow.org/install/source#gpu\n\n'
    count += 1 

    retval += f'{count}. Disable the GPU by defining the environment variable: CUDA_VISIBLE_DEVICES=-1, e.g.:\n'
    if os.name == 'nt':
        retval += '   set CUDA_VISIBLE_DEVICES=-1\n'
    else:
        retval += '   export CUDA_VISIBLE_DEVICES=-1\n'

    return retval
