from typing import TextIO
import logging
import os 
import sys
import re 
import time
import threading

from mltk.utils.path import fullpath
from mltk.utils.shell_cmd import run_shell_cmd
from mltk.utils.serial_reader import SerialReader
from mltk.utils import commander
from mltk.utils.logger import get_logger, make_filelike


def download_image_and_run(
    firmware_image_path:str,
    platform:str = None,
    masserase:bool = False,
    device:str = None,
    serial_number:str = None,
    ip_address:str = None,
    setup_script:str = None,
    setup_script_args:str = None,
    program_script:str=None,
    program_script_args:str = None,
    reset_script:str=None,
    reset_script_args:str = None,
    port:str = None,
    baud:int = None,
    timeout:float = None,
    start_msg:str = None,
    completed_msg:str = None,
    retries:int = 0,
    logger:logging.Logger = None,
    abort_event:threading.Event = None,
    outfile:TextIO=None
):
    """Run a firmware image on a device and parse its serial output for errors
    
    Args:
        firmware_image_path (str): Path to firmware executable to program to device
        platform (str): Platform name, infer from connected device if omitted
        masserase (bool): Mass erase device before programming firmware image
        device (str): JLink device code, infer from platform is omitted
        serial_number (str): J-Link debugger USB serial number
        ip_address (str): J-Link debugger IP address
        setup_script (str): Path to python script to execute before programming device
        setup_script_args (str): Arguments to pass to setup script
        program_script (str): Path to python script to execute to program device. 
            If omitted, then use Commander to program device
        setup_script_args (str): Arguments to pass to setup script
        program_script_args (str): Arguments to pass to program script.
            If the value _IMAGE_PATH_ is in the args string, 
            then it will automatically be replaced with the given firmware_image_path
        reset_script (str): Path to python script to reset device after programming
            If omitted, then use Commander to reset the device
        reset_script_args (str): Arguments to pass to reset script
        port (str): Serial COM port
        baud (int): Serial COM port BAUD
        timeout (float): Maximum time in seconds to wait for program to complete on device
        verbose (bool): Enable verbose logging
        start_msg (str): Regex for app to print to console for it the serial logger to start recording
        completed_msg (str): Regex for app to print to console for it to have successfully completed
        retries (int): The number of times to retry running the firmware on the device
        outfile (TextIO): File to store serial output, if omitted then use provided logger
    """
    abort_event = abort_event or threading.Event()
    logger = logger or get_logger()
    if not outfile:
        make_filelike(logger)
        outfile = logger

    commander.set_adapter_info(
        serial_number=serial_number,
        ip_address=ip_address,
    )
    stop_regex =[re.compile(r'.*done.*', re.IGNORECASE)]
    if completed_msg:
        logger.debug(f'Completed msg regex: {completed_msg}')
        stop_regex.append(completed_msg)

    firmware_image_path = fullpath(firmware_image_path)
    if not os.path.exists(firmware_image_path):
        raise FileNotFoundError(f'Firmware image file not found: {firmware_image_path}')
    

    if setup_script:
        setup_script = fullpath(setup_script)
        if not os.path.exists(setup_script):
            raise FileNotFoundError(f'Invalid setup_script argument, file not found: {setup_script}')
    if program_script:
        program_script = fullpath(program_script)
        if not os.path.exists(program_script):
            raise FileNotFoundError(f'Invalid program_script argument, file not found: {program_script}')
    if reset_script:
        reset_script = fullpath(reset_script)
        if not os.path.exists(reset_script):
            raise FileNotFoundError(f'Invalid reset_script argument, file not found: {reset_script}')
        


    if setup_script:
        setup_script_args = _parse_script_args(setup_script_args if setup_script_args else "")
        setup_cmd = f'{sys.executable} "{setup_script}" {setup_script_args}'
        logger.info(f'Running setup script: {setup_cmd}')
        retcode, _ = run_shell_cmd(setup_cmd, outfile=logger)
        if retcode != 0:
            raise RuntimeError(f'Failed to execute setup script, err={retcode}')
        
    if program_script:
        program_script_args = program_script_args or '_IMAGE_PATH_'
        program_script_args = program_script_args.replace('_IMAGE_PATH_', firmware_image_path)
        program_script_args = _parse_script_args(program_script_args)
        program_cmd = f'{sys.executable} "{program_script}" {program_script_args}'

        logger.info(f'Running program script: {program_cmd}')
        retcode, _ = run_shell_cmd(program_cmd, outfile=logger)
        if retcode != 0:
            raise RuntimeError(f'Failed to execute setup script, err={retcode}')
    else:
        if masserase:
            commander.masserse_device(
                platform=platform,
                device=device
            )

        logger.info(f'Programming {firmware_image_path} to device ...')
        commander.program_flash(
            firmware_image_path,
            platform=platform,
            device=device,
            show_progress=False,
            halt=True,
            logger=logger
        )

    # If no serial COM port is provided,
    # then attemp to resolve it based on common Silab's board COM port description
    port = port or 'regex:JLink CDC UART Port'
    baud = baud or 115200

    max_retries = max(retries, 1)
    for retry_count in range(1, max_retries+1):
        logger.error(f'Executing application on device (attempt {retry_count} of {max_retries}) ...')
        logger.info(f'Opening serial connection, BAUD={baud}, port={port}')
        with SerialReader(
            port=port,
            baud=baud,
            outfile=outfile,
            start_regex=start_msg,
            stop_regex=stop_regex,
            fail_regex=[
                re.compile(r'.*hardfault.*', re.IGNORECASE),
                re.compile(r'.*error.*', re.IGNORECASE),
                re.compile(r'.*failed to alloc memory.*', re.IGNORECASE),
                re.compile(r'.*assert failed.*', re.IGNORECASE)
            ]
        ) as serial_reader:
            # Reset the board to start the profiling firmware
            if reset_script:
                reset_script_args = _parse_script_args(reset_script_args if reset_script_args else "")
                reset_cmd = f'{sys.executable} "{reset_script}" {reset_script_args}'
                logger.info(f'Running reset script: {reset_cmd}')
                retcode, _ = run_shell_cmd(reset_cmd, outfile=logger)
                if retcode != 0:
                    raise RuntimeError(f'Failed to execute reset script, err={retcode}')
            else:
                commander.reset_device(
                    platform=platform,
                    device=device,
                    logger=logger,
                )

            # Wait for up to a minute for the profiler to complete
            # The read() will return when the stop_regex, fail_regex, or timeout condition is met
            if not serial_reader.read(
                activity_timeout=timeout, 
                abort_event=abort_event
            ):
                if abort_event.is_set():
                    logger.info("Application aborted")
                    return 
                    
                logger.error(f'Timed-out ({timeout}s) waiting for app on device to complete')
                if retry_count < max_retries:
                    serial_reader.close()
                    time.sleep(3.0) # Wait a moment and retry
                    continue

                raise TimeoutError(f'Timed-out ({timeout}s) waiting for app on device to complete')

            # Check if the profiler failed
            if serial_reader.failed:
                logger.error(f'App failed on device, err: {serial_reader.error_message}')
                if retry_count < max_retries:
                    serial_reader.close()
                    time.sleep(3.0) # Wait a moment and retry
                    continue

                raise RuntimeError(f'App failed on device, err: {serial_reader.error_message}')

            break

    logger.info('Application successfully executed')



def _parse_script_args(args_str:str) -> str:
    if args_str.startswith('"') and args_str.endswith('"'):
        args_str = args_str[1:-1]

    args_str = args_str.replace('\\ ', ' ')

    return args_str