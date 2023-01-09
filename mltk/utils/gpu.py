"""GPU utilities

See the source code on Github: `mltk/utils/gpu.py <https://github.com/siliconlabs/mltk/blob/master/mltk/utils/gpu.py>`_
"""

import os
import sys
import re
import atexit
import logging
from collections import namedtuple

from .path import get_user_setting

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



def disable():
    """Disable the GPU from being used by Tensorflow"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def initialize(logger=None):
    """Initialize the GPU for usage with Tensorflow

    NOTE: The deinitialize() API will automatically be called when the script exits
    """

    selected_gpus = globals().get('selected_gpus', [])
    if selected_gpus:
        return
    globals()['selected_gpus'] = []


    CUDA_VISIBLE_DEVICES = get_user_setting(
        'cuda_visible_devices',
            os.getenv('MLTK_CUDA_VISIBLE_DEVICES',
                os.getenv('CUDA_VISIBLE_DEVICES', '')
            )
        ).strip()

    logger.debug(f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}')

    if CUDA_VISIBLE_DEVICES == '-1':
        return

    logger = logger or DummyLogger()


    try:
        # %% Select available GPU
        import GPUtil
        import tensorflow as tf

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        gpus = GPUtil.getGPUs()

        if len(gpus) == 0:
            logger.info("No GPUs found, using CPU for training")
            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
            return

        logger.debug(f"Available GPUs:\n" + "\n".join([f"- {g.name} (id={g.id})" for g in gpus]))

        if not CUDA_VISIBLE_DEVICES:
            logger.debug('Searching best GPU available')
            best_gpu = gpus[0]
            for gpu in gpus[1:]:
                if gpu.memoryFree > best_gpu.memoryFree:
                    best_gpu = gpu
            CUDA_VISIBLE_DEVICES = str(best_gpu.id)

        elif CUDA_VISIBLE_DEVICES == 'all':
            logger.debug('Using all available GPUs')
            CUDA_VISIBLE_DEVICES = ','.join(str(x.id) for x in gpus)


        os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

        # Enable dynamic memory growth for GPU
        try:
            tf_gpus = tf.config.list_physical_devices('GPU')
            if len(tf_gpus) == 0:
                _print_warning_msg(logger)
                deinitialize(force=True)
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                return

            gpu_ids = [int(x) for x in CUDA_VISIBLE_DEVICES.split(',')]
            for gpu_id in gpu_ids:
                globals()['selected_gpus'].append(gpu_id)
                for tf_gpu in tf_gpus:
                    if tf_gpu.name.endswith(f'GPU:{gpu_id}'):
                        tf.config.experimental.set_memory_growth(tf_gpu, True)
                        logger.info(f"Selecting GPU : {gpus[gpu_id].name} (id={gpu_id})")

        except Exception as e:
            logger.debug(f'Error configuring GPU(s),  err: {e}')

        # The TfLiteConverter adds a StreamHandler to the root logger,
        # remove it so we don't double print everything to the console
        logging.getLogger().handlers.clear()
        atexit.register(deinitialize)

    except Exception as e:
        err_msg = f'{e}'
        logger.warning(f'GPU init err: {err_msg}')
        if 'Driver/library version mismatch' in err_msg:
            _print_warning_msg(logger)
        logger.info("Using CPU for training")
        deinitialize()
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"



def deinitialize(force=False):
    """De-initialize the GPU

    NOTE: This is automatically called when the script exits
    """
    selected_gpus = globals().get('selected_gpus', [])

    try:
        from numba import cuda

        for gpu_id in selected_gpus:
            cuda.select_device(gpu_id)
            cuda.close()
    except:
        pass
    finally:
        selected_gpus.clear()


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


def _print_warning_msg(logger):
    logger.warning('\n\n\n'
        '*******************************************************************************\n'
        'WARNING: Failed to load GPU driver\n'
        '\n'
        'This could mean that the driver or CUDA libraries are not properly installed,\n'
        'or that your installed GPU driver does not match the Tensorflow version.\n\n'
        'Refer to the Tensorflow GPU installation guide here:\n'
        'https://www.tensorflow.org/install/gpu\n'
        'https://www.tensorflow.org/install/source#gpu\n'
        '\n'
        'Alternatively, you can disable the GPU by defining the environment variable: CUDA_VISIBLE_DEVICES=-1\n'
        '.e.g.:\n'
        f'{"set" if os.name == "nt" else "export"} CUDA_VISIBLE_DEVICES=-1\n\n'
        '*******************************************************************************\n'
    )