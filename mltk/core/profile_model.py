
from typing import Union, List, Tuple
import re
import os
import copy



from mltk.utils.logger import get_logger, make_filelike
from mltk.utils.path import fullpath
from mltk.utils.python import append_exception_msg

from .model import (MltkModel, load_tflite_model)
from .tflite_model import TfliteModel
from .profiling_results import ProfilingModelResults, ProfilingLayerResult
from .utils import (get_mltk_logger, ArchiveFileNotFoundError)
from .tflite_micro import TfliteMicro, TfliteMicroLayerError
from .tflite_model_parameters import TfliteModelParameters


def profile_model(
    model:Union[MltkModel, TfliteModel, str], 
    accelerator:str=None,
    port:str=None,
    use_device:bool=False,
    build:bool=False, 
    **kwargs
) -> ProfilingModelResults:
    """Profile a model for the given accelerator
    
    This will profile the given model in either a
    hardware simulator or on a physical device.

    Refer to the `Model Profiler <https://siliconlabs.github.io/mltk/docs/guides/model_profiler.html>`_ guide for more details.

    Args:
        model: The model to profile as either a :py:class:`mltk.core.MltkModel` or :py:class:`mltk.core.TfliteModel` instance,
            or the path to a `.tflite` or `.mltk.zip` file
        accelerator: The name of the hardware accelerator to profile for If omitted, then use reference kernels
        use_device: Profile on a locally connected embedded device. If omitted, then profile in simulator
        port: Serial port of physical platform. If omitted, attempt to discover automatically
        build: If true, build the MLTK Model as a .tflite before profiling

    Returns:
        The results of model profiling
    """
    
    accelerator = TfliteMicro.normalize_accelerator_name(accelerator)

    try:
        tflite_model = load_tflite_model(model=model, build=build)
    except ArchiveFileNotFoundError as e:
        append_exception_msg(e,
            '\nAlternatively, add the --build option to profile the model without training it first'
        )
        raise


    if use_device:
        # Profile on embedded device
        profiling_model_results = profile_model_on_device(
            tflite_model, 
            accelerator=accelerator, 
            port=port
        )
    else:
        # Profile in hardware simulator
        profiling_model_results = profile_model_in_simulator(
            tflite_model, 
            accelerator=accelerator,
            **kwargs
        )

    profiling_model_results._model_name = (tflite_model.filename or 'my_model.tflite')[:-len('.tflite')] # pylint: disable=protected-access

    return profiling_model_results


def profile_model_in_simulator(
    tflite_model:TfliteModel,
    accelerator:str=None,
    **kwargs
) -> ProfilingModelResults:
    """Profile the given TfliteModel in simulator
    
    Args:
        tflite_model: .tflite model to profile
        accelerator: Optional, name of accelerator to profile for

    Returns:
        ProfilingModelResults: Results of the model profiling
    """

    logger = get_mltk_logger()
    logger.error('Profiling model in simulator ...')

    # Profile the model in the hardware simulator
    # and estimate various metrics if possible
    profiling_results = TfliteMicro.profile_model(
        tflite_model,
        accelerator=accelerator,
        return_estimates=True,
        **kwargs
    )

    return profiling_results


def profile_model_on_device(
    tflite_model:TfliteModel,
    accelerator:str=None,
    port:str=None
) -> ProfilingModelResults:
    """Profile the given TfliteModel on a physical embedded target
    
    Args:
        tflite_model: TfliteModel instance
        accelerator: Name of hardware accelerator
        port: Serial COM port

    Returns:
        ProfilingModelResults: Results of the model profiling
    """
    # pylint: disable=import-outside-toplevel
    from mltk.utils import commander
    from mltk.utils import firmware_apps
    from mltk.utils.serial_reader import SerialReader

    logger = get_mltk_logger()
    accelerator = TfliteMicro.normalize_accelerator_name(accelerator)
    tflite_model = copy.deepcopy(tflite_model)
    try:
        tflite_model_params = TfliteModelParameters.load_from_tflite_model(tflite_model)
        if 'runtime_memory_size' in tflite_model_params:
            del tflite_model_params['runtime_memory_size'] # Ensure the memory size is -1 so it is calculated at runtime
        tflite_model_params.add_to_tflite_model(tflite_model)
    except:
        # If the model doesn't have params then just ignore the error
        pass

    logger.error('Programming ML model to device ...')
    firmware_apps.program_image_with_model(
        name='mltk_model_profiler',
        accelerator=accelerator,
        tflite_model=tflite_model,
        logger=logger,
        halt=True
    )

    # If no serial COM port is provided, 
    # then attemp to resolve it based on common Silab's board COM port description
    port = port or 'regex:JLink CDC UART Port'
    
    # We want the serial logger to always write to the file
    # but only to the console if verbose logging is enabled
    serial_logger = get_logger('serial_logger', 'DEBUG', parent=logger)
    make_filelike(serial_logger, level='INFO' if logger.verbose else 'DEBUG')
       
    # Start the serial COM port reader
    logger.error('Profiling ML model on device ...')
    with SerialReader( 
        port=port,
        baud=115200, 
        outfile=serial_logger,
        start_regex=[
            re.compile('.*Starting Model Profiler', re.IGNORECASE), 
            re.compile('Loading model', re.IGNORECASE)
        ],
        stop_regex=[re.compile(r'.*done.*', re.IGNORECASE)],
        fail_regex=[
            re.compile(r'.*hardfault.*', re.IGNORECASE), 
            re.compile(r'.*error.*', re.IGNORECASE),
            re.compile(r'.*failed to alloc memory.*', re.IGNORECASE)
        ]
    ) as serial_reader:
        # Reset the board to start the profiling firmware
        commander.reset_device()

        # Wait for up to a minute for the profiler to complete
        if not serial_reader.read(timeout=60):
            raise TimeoutError('Timed-out waiting for profiler on device to complete')

        # Check if the profiler failed
        if serial_reader.failed:
            raise RuntimeError(f'Profiler failed on device, err: {serial_reader.error_message}')

        # Retrieve the captured data
        device_log = serial_reader.captured_data

    return parse_device_model_profiler_log(
        device_log,
        tflite_model=tflite_model,
        accelerator=accelerator
    )



def parse_device_model_profiler_log(
    log_data:str,
    tflite_model:TfliteModel,
    accelerator:str
) -> ProfilingModelResults:
    # pylint: disable=protected-access
    lines = [x.strip() for x in log_data.splitlines()]

    runtime_memory_size = 0
    cpu_clock_rate = 0
    layer_results:List[ProfilingLayerResult] = []
    
    cpu_clock_re = re.compile(r'CPU clock:\s([\d\.Mk]+)Hz')
    runtime_memory_re = re.compile(r'Tensor runtime memory:\s([\d\.Mk]+)')
    layer_name_re = re.compile(r'Op(\d+)-\S+')
    cpu_cycles_re = re.compile(r'([\d\.kMG]+) CPU cycles')
    acc_cycles_re = re.compile(r'([\d\.kMG]+) Accelerator cycles')
    ops_cycles_re = re.compile(r'([\d\.kMG]+) OPs')
    macs_cycles_re = re.compile(r'([\d\.kMG]+) MACs')
    error_msg_re = TfliteMicroLayerError.WARNING_RE
    time_ms_re = re.compile(r'([\d\.]+) ms')

    layer_error_msgs = {}
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        line_index += 1

        match = cpu_clock_re.match(line)
        if match:
            cpu_clock_rate = _line_to_int(match.group(1))
            continue

        match = runtime_memory_re.match(line)
        if match:
            runtime_memory_size = _line_to_int(match.group(1))
            continue

        match = error_msg_re.match(line)
        if match:
            layer_error_msgs[int(match.group(1))] = match.group(3)
            continue

        match = layer_name_re.match(line)
        if not match:
            continue

        layer_index = int(match.group(1))
        i = len(layer_results)
        while i <= layer_index:
            layer_err_msg = None if i not in layer_error_msgs else layer_error_msgs[i]
            layer_results.append(ProfilingLayerResult(
                tflite_layer=tflite_model.layers[i],
                error_msg=layer_err_msg
            ))
            i += 1

        layer = layer_results[layer_index]
        while line_index < len(lines):
            line = lines[line_index]
            match = layer_name_re.match(line)
            if match or line.startswith('--') or not line:
                break
            line_index += 1

            match = ops_cycles_re.match(line)
            if match:
                layer._ops = _line_to_int(match.group(1))
                continue
            match = macs_cycles_re.match(line)
            if match:
                layer._macs = _line_to_int(match.group(1))
                continue
            match = cpu_cycles_re.match(line)
            if match:
                layer._cpu_cycles = _line_to_int(match.group(1))
                continue
            match = acc_cycles_re.match(line)
            if match:
                layer._accelerator_cycles = _line_to_int(match.group(1))
                continue
            match = time_ms_re.match(line)
            if match:
                layer._time = float(match.group(1))/1e3
                continue


    return ProfilingModelResults(
        model=tflite_model,
        accelerator=accelerator,
        cpu_clock_rate=cpu_clock_rate,
        runtime_memory_bytes=runtime_memory_size,
        layers=layer_results,
        is_simulated=False
    )


def _line_to_int(line:str) -> int:
    multiplier = 1
    if 'k' in line:
        multiplier = 1e3
    elif 'M' in line:
        multiplier = 1e6
    elif 'G' in line:
        multiplier = 1e9
    if multiplier > 1:
        line = line[:-1]
    v = float(line)
    return int(v * multiplier)

