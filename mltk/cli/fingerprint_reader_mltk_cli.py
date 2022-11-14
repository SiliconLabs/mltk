import os 
import sys
import re
import time
import threading
import collections
import logging
import signal
import typer
import atexit

from mltk import cli



@cli.root_cli.command('fingerprint_reader')
def fingerprint_reader_command(
    model: str = typer.Argument(..., 
        help='''\b
On of the following:
- MLTK model name 
- Path to .tflite file
- Path to model archive file (.mltk.zip)
NOTE: The model must have been previously trained for image classification''',
        metavar='<model>'
    ),
    accelerator: str = typer.Option(None, '--accelerator', '-a',
        help='''\b
Name of accelerator to use while executing the audio classification ML model''',
        metavar='<name>'
    ),
    port:str = typer.Option(None,
        help='''\b
Serial COM port of a locally connected embedded device.
'If omitted, then attempt to automatically determine the serial COM port''',
        metavar='<port>'
    ),
    verbose: bool = typer.Option(False, '--verbose', '-v', 
        help='Enable verbose console logs'
    ),
    generate_dataset: bool = typer.Option(False, '--generate-dataset', '-g', 
        help='''\b
Generate a fingerprint dataset by guiding the user through a sequence of finger captures and saving to the PC.
NOTE: With this option, --no-inference is automatically enabled
'''
    ),
    samples_per_finger: int = typer.Option(5, '--samples-per-finger', '-c', 
        help='The number of samples per finger to collect when generating the dataset'
    ),
    dataset_dir: str = typer.Option(None,
        help='Base directory where dataset should be generated'
    ),
    dump_images: bool = typer.Option(False, '--dump-images', '-d', 
        help='''\b
Dump the raw images from the device fingerprint reader to a directory on the local PC. 
''',
    ),
    disable_inference: bool = typer.Option(None,  '--no-inference', '-z',
        help='By default inference is executed on the device. Use --no-inference to disable inference on the device which can improve dumping throughput',
        is_flag=True
    ),
    app_path: str = typer.Option(None, '--app',
        help='''\b
By default, the fingerprint_authenticator app is automatically downloaded. 
This option allows for overriding with a custom built app.
Alternatively, set this option to "none" to NOT program the fingerprint_authenticator app to the device.
In this case, ONLY the .tflite will be programmed and the existing fingerprint_authenticator app will be re-used.
''',
        metavar='<path>'
    ),
    is_unit_test: bool = typer.Option(False, '--test', 
        help='Run as a unit test',
    ),
):
    """View/save fingerprints captured by the fingerprint eader connected to an embedded device.

    \b
    NOTE: A supported embedded device must be locally connected to use this command.
    Additionally, an R503 fingerprint module: https://www.adafruit.com/product/4651
    must be connected connected to the development board.
    Refer to the online documentation for how to connect it to the development:
    https://siliconlabs.github.io/mltk/docs/cpp_development/examples/fingerprint_authenticator.html#hardware-setup
    """
    # Import all required packages here instead of at top
    # to help improve the CLI's responsiveness

    import numpy as np
    from mltk.core import (
        TfliteModelParameters,
        load_tflite_model,
    )

    from mltk.utils import firmware_apps
    from mltk.utils import commander
    from mltk.utils.serial_reader import SerialReader
    from mltk.utils.path import (create_user_dir, clean_directory)
    from mltk.utils.jlink_stream import (JlinkStream, JLinkDataStream, JlinkStreamOptions)
    from mltk.utils.logger import get_logger
    from mltk.utils.system import send_signal
    from mltk.utils.string_formatting import iso_time_filename_str


    logger = cli.get_logger()
    latest_image_q = collections.deque(maxlen=1)

    try:
        from cv2 import cv2
    except:
        try:
            import cv2
        except:
            raise RuntimeError('Failed import cv2 Python package, try running: pip install opencv-python OR pip install silabs-mltk[full]')

    if generate_dataset:
        disable_inference = True
    if disable_inference:
        logger.warning('Disabling inference on the device')

    accelerator = cli.parse_accelerator_option(accelerator)
    platform = commander.query_platform()

    try:
        tflite_model = load_tflite_model(
            model,  
            print_not_found_err=True,
            logger=logger
        )
    except Exception as e:
        cli.handle_exception('Failed to load model', e)

    

    ###############################################################
    def _start_ctrl_c_timer():
        """This is used for a unit test to simulate issuing CTRL+C """
        def _on_timeout():
            logger = cli.get_logger()
            logger.warning('Issuing CTRL+C\n')
            send_signal(signal.SIGINT)
        t = threading.Timer(7, _on_timeout)
        t.start()


    ###############################################################
    def _update_model_parameters(): 
        """Update the .tflite embedded model parameters based on the command-line options"""
        params = TfliteModelParameters.load_from_tflite_model(tflite_model)

        if not (verbose or
            disable_inference is not None
        ):
            return params


        if disable_inference:
            params['disable_inference'] = disable_inference

        params.add_to_tflite_model(tflite_model)





    ###############################################################
    def _start_jlink_processor(
        dump_image_dir:str
    ) -> threading.Event:
        """Start the JLink stream interface
       
        This allows for reading binary data from the embedded device via debug interface
        """

        jlink_logger = get_logger('jlink', console=False, parent=logger)

        jlink_logger.debug('Opening device data stream ...')
        opts = JlinkStreamOptions()
        opts.polling_period=0.100

        jlink_stream = JlinkStream(opts)
        jlink_stream.connect()
        jlink_logger.debug('Device data stream opened')

        stop_event = threading.Event()
        atexit.register(stop_event.set)
        t = threading.Thread(
            name='JLink Processing loop', 
            target=_jlink_processing_loop,
            daemon=True,
            kwargs=dict( 
                jlink_stream=jlink_stream,
                stop_event=stop_event, 
                logger=jlink_logger,
                dump_image_dir=dump_image_dir
            )
        )
        t.start()
        return stop_event


    ###############################################################
    def _jlink_processing_loop(
        jlink_stream:JlinkStream, 
        stop_event:threading.Event,
        dump_image_dir:str, 
       
        logger:logging.Logger
    ):
        """Read binary data from embedded device via JLink interface
        
        This runs in a separate thread
        """
        image_stream:JLinkDataStream = None 
        image_width = 192
        image_height = 192
        image_length = image_width*image_height

        image_data = bytearray()
        while True:
            if stop_event.wait(0.010):
                jlink_stream.disconnect()
                break 

            if image_stream is None:
                try:
                    image_stream = jlink_stream.open('raw', mode='r')
                    logger.debug('Device image stream ready')
                except Exception as e:
                    logger.debug(f'Failed to open device image stream, err: {e}')
                    continue

            remaining_length = image_length - len(image_data)
            img_bytes = image_stream.read_all(
                remaining_length, 
                timeout=0.500,
                throw_exception=False
            )
            if img_bytes:
                image_data.extend(img_bytes)
        
            if len(image_data) != image_length:
                continue

            img_buffer = np.frombuffer(image_data, dtype=np.uint8)
            image_data = bytearray()
            img = np.reshape(img_buffer, (image_height, image_width, 1))

            latest_image_q.append(img)

            if dump_image_dir:
                image_path = f'{dump_image_dir}/{iso_time_filename_str()}.jpg'
                cv2.imwrite(image_path, img)
        
            cv2.imshow('Fingerprint', img)
            cv2.waitKey(10)




    ###############################################################
    def _start_dataset_collection_processor(
        dataset_dir:str,
    ) -> threading.Event:
        stop_event = threading.Event()
        atexit.register(stop_event.set)
        t = threading.Thread(
            name='Dataset collection loop', 
            target=_dataset_collection_loop,
            daemon=True,
            kwargs=dict( 
                stop_event=stop_event, 
                dataset_dir=dataset_dir,
            )
        )
        t.start()
        return stop_event


    ###############################################################
    def _dataset_collection_loop(
        stop_event: threading.Event,
        dataset_dir:str,
    ):
        """Collect dataset image by displaying message to the console 
        and receiving feedback via keyword
        """
        finger_names = ['THUMB', 'INDEX Finger', 'MIDDLE Finger', 'RING finger', 'PINKY finger']
        hand_names = ['LEFT', 'RIGHT']

        logger.info('#' * 100)
        logger.info(f'Generating fingerprint dataset at: {dataset_dir}')
        logger.info('Press CTRL+C to stop\n')
       
        while True:
            if stop_event.is_set():
                return
            logger.info('\n--------------------------------------------')
            logger.info(f'Starting collection for new person')
            logger.info('Enter their initials then press ENTER')
            logger.info('NOTE: If the same person is adding new samples then re-enter their initials.')
            logger.info('      The old samples will NOT be erased (CTRL+C to exit)')
            person_id = None
            while not person_id:
                try:
                    if stop_event.is_set():
                        return 
                    person_id = sys.stdin.readline().strip().lower()
                except:
                    return

            for hand_name in hand_names:
                for finger_name in finger_names:
                    logger.info('\n***')
                    logger.info(f'Collecting {samples_per_finger} samples for your {hand_name} hand, {finger_name}')
                    for i in range(samples_per_finger): 
                        logger.info(f'[{i+1} of {samples_per_finger}] Place {hand_name} hand, {finger_name} on fingerprint reader (CTRL+C to exit)')
                        
                        while not latest_image_q:
                            if stop_event.is_set():
                                return 
                            time.sleep(0.1)
                        img = latest_image_q.pop()

                        img_dir= f'{person_id}/{hand_name}/{finger_name.split()[0]}'.lower()
                        img_dir = f'{dataset_dir}/{img_dir}'
                        os.makedirs(img_dir, exist_ok=True)
                        image_path = f'{img_dir}/raw_{iso_time_filename_str()}.jpg'
                        cv2.imwrite(image_path, img)
                        logger.info(f'Saved {image_path}')



    ##################################################################################
    #
    # Actual command logic
    #

    _update_model_parameters()
    
    dump_image_dir = None 
    if dump_images:
        dump_image_dir = create_user_dir(f'fingerprint_reader/{platform}/dump')
        logger.info(f'Dumping fingerprints to {dump_image_dir}')
        clean_directory(dump_image_dir)

    if generate_dataset:
        dataset_dir = dataset_dir or create_user_dir(f'fingerprint_reader/dataset')
        logger.info(f'Generating dataset at {dataset_dir}')


    # Program the image_classifier app and .tflite model
    # to the device's flash
    firmware_apps.program_image_with_model(
        name='mltk_fingerprint_authenticator',
        platform=platform,
        accelerator=accelerator,
        tflite_model=tflite_model,
        logger=logger,
        halt=True,
        firmware_image_path=app_path,
        model_offset=0xC000 # This is the size of the NVM section that is always placed at the end of flash. 
                            # We program the model just before the NVM section
    )

    # If no serial COM port is provided, 
    # then attemp to resolve it based on common Silab's board COM port description
    port = port or 'regex:JLink CDC UART Port'

    # Start the serial COM port reader
    logger.info('Running fingerprint authenticator on device ...')
    logger.info('Press CTRL+C to exit\n')

    with SerialReader( 
        port=port,
        baud=115200, 
        outfile=logger,
        start_regex=re.compile(r'.*app starting.*', re.IGNORECASE),
        fail_regex=[
            re.compile(r'.*hardfault.*', re.IGNORECASE), 
            re.compile(r'.*assert.*', re.IGNORECASE), 
        ]
    ) as reader:
        commander.reset_device()
        if is_unit_test:
            _start_ctrl_c_timer()
        
        stop_jlink_event = None
        stop_dataset_event = None
        # Wait for the device to be ready
        while True:
            reader.read(timeout=0.10)
            # Check if any errors ocurred
            if reader.error_message:
                raise RuntimeError(f'Device error: {reader.error_message}')
            if reader.started:
                break

        stop_jlink_event = _start_jlink_processor(
            dump_image_dir=dump_image_dir
        )
        if dataset_dir is not None:
            stop_dataset_event = _start_dataset_collection_processor(
                dataset_dir=dataset_dir
            )

        try:
            while not reader.read(timeout=.010):
                time.sleep(0.100)
                        
            if reader.error_message:
                if stop_jlink_event: 
                    stop_jlink_event.set()
                raise RuntimeError(f'Device error: {reader.error_message}')

        except KeyboardInterrupt:
            pass 
        finally:
            if stop_jlink_event: 
                stop_jlink_event.set()
            if stop_dataset_event:
                stop_dataset_event.set()