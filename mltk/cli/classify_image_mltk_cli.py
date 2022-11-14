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


@cli.root_cli.command('classify_image')
def classify_image_command(
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
Name of accelerator to use while executing the image classification ML model''',
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
    average_window_duration_ms: int = typer.Option(None, '--window_duration', '-w', 
        help='''\b
Controls the smoothing. Drop all inference results that are older than <now> minus window_duration.
Longer durations (in milliseconds) will give a higher confidence that the results are correct, but may miss some images''',
        metavar='<duration ms>'
    ),
    minimum_count: int = typer.Option(None, '--count', '-c', 
        help='The *minimum* number of inference results to average when calculating the detection value',
        metavar='<count>'
    ),
    detection_threshold: int = typer.Option(None, '--threshold', '-t', 
        help='Minimum averaged model output threshold for a class to be considered detected, 0-255. Higher values increase precision at the cost of recall',
        metavar='<threshold>'
    ),
    suppression_count: int = typer.Option(None, '--suppression', '-s', 
        help='Number of samples that should be different than the last detected sample before detecting again',
        metavar='<count>'
    ),
    latency_ms: int = typer.Option(None, '--latency', '-l', 
        help='This the amount of time in milliseconds between processing loops',
        metavar='<latency ms>'
    ),
    sensitivity: float = typer.Option(None, '--sensitivity', '-i', 
        help='Sensitivity of the activity indicator LED. Much less than 1.0 has higher sensitivity',
    ),
    dump_images: bool = typer.Option(False, '--dump-images', '-x', 
        help='''\b
Dump the raw images from the device camera to a directory on the local PC. 
NOTE: Use the --no-inference option to ONLY dump images and NOT run inference on the device
Use the --dump-threshold option to control how unique the images must be to dump
''',
    ),
    dump_threshold: float = typer.Option(0.1,
        help='''\b
This controls how unique the camera images must be before they're dumped.
This is useful when generating a dataset.
If this value is set to 0 then every image from the camera is dumped.
if this value is closer to 1. then the images from the camera should be sufficiently unique from
prior images that have been dumped.
''',
    ),
    disable_inference: bool = typer.Option(None,  '--no-inference',
        help='By default inference is executed on the device. Use --no-inference to disable inference on the device which can improve image dumping throughput',
        is_flag=True
    ),
    app_path: str = typer.Option(None, '--app',
        help='''\b
By default, the image_classifier app is automatically downloaded. 
This option allows for overriding with a custom built app.
Alternatively, set this option to "none" to NOT program the image_classifier app to the device.
In this case, ONLY the .tflite will be programmed and the existing image_classifier app will be re-used.
''',
        metavar='<path>'
    ),
    is_unit_test: bool = typer.Option(False, '--test', 
        help='Run as a unit test',
    ),
):
    """Classify images detected by a camera connected to an embedded device.

    \b
    NOTE: A supported embedded device must be locally connected to use this command.
    Additionally, an Arducam camera module:
    https://www.arducam.com/product/arducam-2mp-spi-camera-b0067-arduino
    must be connected to the development board.
    Refer to the online documentation for how to connect it to the development board:
    https://siliconlabs.github.io/mltk/docs/cpp_development/examples/image_classifier.html#hardware-setup


    \b
    ----------
     Examples
    ----------
    \b
    # Classify images using the rock_paper_scissors model
    # Verbosely print the inference results
    mltk classify_image rock_paper_scissors --verbose 
    \b
    # Classify images using the rock_paper_scissors model
    # using the MVP hardware accelerator
    mltk classify_image rock_paper_scissors --accelerator MVP 
    \b
    # Classify images using the rock_paper_scissors model
    # and dump the images to the local PC
    mltk classify_image rock_paper_scissors --dump-images --dump-threshold 0.1

    """
    # Import all required packages here instead of at top
    # to help improve the CLI's responsiveness

    import numpy as np
    from mltk.core import (
        TfliteModelParameters,
        load_tflite_model
    )
    from mltk.core.preprocess.image.image_database import UniqueImageDatabase

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
            average_window_duration_ms is not None or
            detection_threshold or
            suppression_count is not None or
            minimum_count or
            disable_inference or 
            sensitivity or
            latency_ms is not None
        ):
            return params

        if verbose:
            params['verbose_inference_output'] = True
        if average_window_duration_ms:
            params['average_window_duration_ms'] = average_window_duration_ms
        if detection_threshold:
            params['detection_threshold'] = detection_threshold
        if suppression_count:
            params['suppression_count'] = suppression_count
        if minimum_count:
            params['minimum_count'] = minimum_count
        if disable_inference:
            params['enable_inference'] = False
        if sensitivity:
            params['activity_sensitivity'] = sensitivity
        if latency_ms is not None:
            params['latency_ms'] = latency_ms

        params.add_to_tflite_model(tflite_model)





    ###############################################################
    def _start_jlink_processor(
        dump_image_dir:str
    ) -> threading.Event:
        """Start the JLink stream inferface
       
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
        _, height, width, channels = tflite_model.inputs[0].shape
        image_length = height*width*channels
        image_database = UniqueImageDatabase(
            maxlen=128,
            threshold=dump_threshold,
            quadrants=2
        )

        image_data = bytearray()
        while True:
            if stop_event.wait(0.010):
                jlink_stream.disconnect()
                break 

            if image_stream is None:
                try:
                    image_stream = jlink_stream.open('image', mode='r')
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
            if channels == 1:
                img = np.reshape(img_buffer, (height, width))
            else:
                img = np.reshape(img_buffer, (height, width, channels))

            latest_image_q.append(img)

            if dump_image_dir:
                # Only dump the image if it is unique
                if image_database.add(img):
                    image_path = f'{dump_image_dir}/{iso_time_filename_str()}.jpg'
                    cv2.imwrite(image_path, img)
            
            cv2.imshow('Development Board Samples', _resize_jpg_image(img, 240, 640))
            cv2.waitKey(10)



    ###############################################################
    def _resize_jpg_image(im:np.ndarray, min_dim:int, max_dim:int) -> np.ndarray:
        """Resize an image based on the given dimensions"""
        h, w = im.shape[:2]

        if h < w:
            if h < min_dim:
                resize_h = min_dim
                resize_w = int(resize_h * w / h)
            elif w > max_dim:
                resize_w = max_dim
                resize_h = int(resize_w * h / w)
            else:
                return im
        else:
            if w < min_dim:
                resize_w = min_dim
                resize_h = int(resize_w * h / w)
            elif h > max_dim:
                resize_h = max_dim
                resize_w = int(resize_h * w / h)
            else:
                return im

        return cv2.resize(im, (resize_h, resize_w))



    ##################################################################################
    #
    # Actual command logic
    #

    _update_model_parameters()

    dump_image_dir = None 
    if dump_images:
        dump_image_dir = create_user_dir(f'image_classifier_images/{platform}/dump')
        logger.info(f'Dumping images to {dump_image_dir}')
        clean_directory(dump_image_dir)


    # Program the image_classifier app and .tflite model
    # to the device's flash
    firmware_apps.program_image_with_model(
        name='mltk_image_classifier',
        platform=platform,
        accelerator=accelerator,
        tflite_model=tflite_model,
        logger=logger,
        halt=True,
        firmware_image_path=app_path
    )

    # If no serial COM port is provided, 
    # then attemp to resolve it based on common Silab's board COM port description
    port = port or 'regex:JLink CDC UART Port'

    # Start the serial COM port reader
    logger.info('Running image classifier on device ...')
    logger.info('Press CTRL+C to exit\n')

    with SerialReader( 
        port=port,
        baud=115200, 
        outfile=logger,
        start_regex=re.compile(r'.*Image Classifier.*', re.IGNORECASE),
        fail_regex=[
            re.compile(r'.*hardfault.*', re.IGNORECASE), 
            re.compile(r'.*assert.*', re.IGNORECASE), 
            re.compile(r'.*error.*', re.IGNORECASE)
        ]
    ) as reader:
        commander.reset_device()
        if is_unit_test:
            _start_ctrl_c_timer()
        
        stop_jlink_event = None
        if dump_image_dir:
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

        try:
            while not reader.read(timeout=.010):
                time.sleep(0.100)
                        
            if reader.error_message:
                if stop_jlink_event: 
                    stop_jlink_event.set()
                raise RuntimeError(f'Device error: {reader.error_message}')

        except KeyboardInterrupt:
            if stop_jlink_event: 
                stop_jlink_event.set()