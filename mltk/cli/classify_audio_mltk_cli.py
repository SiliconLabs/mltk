import os
import re
import logging
import wave
import time
import threading
import signal
import atexit
import functools
import collections

import typer
import numpy as np


from mltk import cli


@cli.root_cli.command('classify_audio')
def classify_audio_command(
    model: str = typer.Argument(..., 
        help='''\b
On of the following:
- MLTK model name 
- Path to .tflite file
- Path to model archive file (.mltk.zip)
NOTE: The model must have been previously trained for keyword spotting''',
        metavar='<model>'
    ),
    accelerator: str = typer.Option(None, '--accelerator', '-a',
        help='''\b
Name of accelerator to use while executing the audio classification ML model.
If omitted, then use the reference kernels
NOTE: It is recommended to NOT use an accelerator if running on the PC since the HW simulator can be slow.''',
        metavar='<name>'
    ),
    use_device:bool = typer.Option(False, '-d', '--device',
        help='''\b
If provided, then run the keyword spotting model on an embedded device, otherwise use the PC's local microphone.
If this option is provided, then the device must be locally connected'''
    ),
    port:str = typer.Option(None,
        help='''\b
Serial COM port of a locally connected embedded device.
This is only used with the --device option.
'If omitted, then attempt to automatically determine the serial COM port''',
        metavar='<port>'
    ),
    verbose: bool = typer.Option(False, '--verbose', '-v', 
        help='Enable verbose console logs'
    ),
    average_window_duration_ms: int = typer.Option(None, '--window_duration', '-w', 
        help='''\b
Controls the smoothing. Drop all inference results that are older than <now> minus window_duration.
Longer durations (in milliseconds) will give a higher confidence that the results are correct, but may miss some commands''',
        metavar='<duration ms>'
    ),
    minimum_count: int = typer.Option(None, '--count', '-c', 
        help='The *minimum* number of inference results to average when calculating the detection value. Set to 0 to disable averaging',
        metavar='<count>'
    ),
    detection_threshold: int = typer.Option(None, '--threshold', '-t', 
        help='Minimum averaged model output threshold for a class to be considered detected, 0-255. Higher values increase precision at the cost of recall',
        metavar='<threshold>'
    ),
    suppression_ms: int = typer.Option(None, '--suppression', '-s', 
        help='Amount of milliseconds to wait after a keyword is detected before detecting new keywords',
        metavar='<suppression ms>'
    ),
    latency_ms: int = typer.Option(None, '--latency', '-l', 
        help='This the amount of time in milliseconds between processing loops',
        metavar='<latency ms>'
    ),
    microphone: str = typer.Option(None, '--microphone', '-m', 
        help='For non-embedded, this specifies the name of the PC microphone to use',
        metavar='<name>'
    ),
    volume_gain: int = typer.Option(None, '--volume', '-u', 
        help='Set the volume gain scaler (i.e. amplitude) to apply to the microphone data. If 0 or omitted, no scaler is applied',
        metavar='<volume gain>'
    ),
    dump_audio: bool = typer.Option(False, '--dump-audio', '-x', 
        help='Dump the raw microphone and generate a corresponding .wav file',
    ),
    dump_raw_spectrograms: bool = typer.Option(False, '--dump-raw-spectrograms', '-w', 
        help='Dump the raw (i.e. unquantized) generated spectrograms to .jpg images and .mp4 video',
    ),
    dump_quantized_spectrograms: bool = typer.Option(False, '--dump-spectrograms', '-z', 
        help='Dump the quantized generated spectrograms to .jpg images and .mp4 video',
    ),
    sensitivity: float = typer.Option(None, '--sensitivity', '-i', 
        help='Sensitivity of the activity indicator LED. Much less than 1.0 has higher sensitivity',
    ),
    app_path: str = typer.Option(None, '--app',
        help='''\b
By default, the audio_classifier app is automatically downloaded. 
This option allows for overriding with a custom built app.
Alternatively, if using the --device option, set this option to "none" to NOT program the audio_classifier app to the device.
In this case, ONLY the .tflite will be programmed and the existing audio_classifier app will be re-used.
''',
        metavar='<path>'
    ),
    is_unit_test: bool = typer.Option(False, '--test', 
        help='Run as a unit test',
    ),
):
    """Classify keywords/events detected in a microphone's streaming audio

    NOTE: This command is experimental. Use at your own risk!
    
    \b
    This command runs an audio classification application on either the local PC OR
    on an embedded target. The audio classification application loads the given 
    audio classification ML model (e.g. Keyword Spotting) and streams real-time audio
    from the local PC's/embedded target's microphone into the ML model.

    \b
    System Dataflow:
    Microphone -> AudioFeatureGenerator -> ML Model -> Command Recognizer -> Local Terminal  
    \b
    Refer to the mltk.models.tflite_micro.tflite_micro_speech model for a reference on how to train
    an ML model that works the audio classification application.
    \b
    ----------
     Examples
    ----------
    \b
    # Classify audio on local PC using tflite_micro_speech model   
    # Simulate the audio loop latency to be 200ms  
    # i.e. If the app was running on an embedded target, it would take 200ms per audio loop  
    # Also enable verbose logs  
    mltk classify_audio tflite_micro_speech --latency 200 --verbose 

    \b
    # Classify audio on an embedded target using model: ~/workspace/my_model.tflite   
    # and the following classifier settings:  
    # - Set the averaging window to 1200ms (i.e. drop samples older than <now> minus window)  
    # - Set the minimum sample count to 3 (i.e. must have at last 3 samples before classifying)  
    # - Set the threshold to 175 (i.e. the average of the inference results within the averaging window must be at least 175 of 255)  
    # - Set the suppression to 750ms (i.e. Once a keyword is detected, wait 750ms before detecting more keywords)  
    # i.e. If the app was running on an embedded target, it would take 200ms per audio loop  
    mltk classify_audio /home/john/my_model.tflite --device --window 1200ms --count 3 --threshold 175 --suppression 750  

    \b
    # Classify audio and also dump the captured raw audio and spectrograms  
    mltk classify_audio tflite_micro_speech --dump-audio --dump-spectrograms
    
    """

    # Import all required packages here instead of at top
    # to help improve the CLI's responsiveness

    from mltk.core import (
        TfliteModelParameters,
        load_tflite_model,
    )

    from mltk.utils import firmware_apps
    from mltk.utils import commander
    from mltk.utils.system import (get_current_os, make_path_executable, send_signal)
    from mltk.utils.shell_cmd import run_shell_cmd
    from mltk.utils.serial_reader import SerialReader
    from mltk.utils.path import (create_tempdir, create_user_dir, clean_directory)
    from mltk.utils.jlink_stream import (JlinkStream, JLinkDataStream, JlinkStreamOptions)
    from mltk.utils.logger import get_logger


    logger = cli.get_logger()

    try:
        from cv2 import cv2
    except Exception:
        try:
            import cv2
        except:
            raise RuntimeError('Failed import cv2 Python package, try running: pip install opencv-python OR pip install silabs-mltk[full]')

    accelerator = cli.parse_accelerator_option(accelerator)

    try:
        tflite_model = load_tflite_model(
            model, 
            print_not_found_err=True,
            logger=logger
        )
    except Exception as e:
        cli.handle_exception('Failed to load model', e)

    input_dtype = tflite_model.inputs[0].dtype
    platform = get_current_os() if not use_device else commander.query_platform()
  

    ###############################################################
    def _run_audio_classifier_on_device( 
        tflite_model_params:TfliteModelParameters,
        dump_audio_dir:str, 
        dump_raw_spectrograms_dir:str,
        dump_quantized_spectrograms_dir:str,
    ):
        """Run the audio_classifier app on an embedded device"""
        nonlocal port

        # Program the audio_classifier app and .tflite model
        # to the device's flash
        firmware_apps.program_image_with_model(
            name='mltk_audio_classifier',
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
        logger.info('Running audio classifier on device ...')
        logger.info('Press CTRL+C to exit\n')

        with SerialReader( 
            port=port,
            baud=115200, 
            outfile=logger,
            start_regex=re.compile(r'.*Audio Classifier.*', re.IGNORECASE),
            fail_regex=[
                re.compile(r'.*hardfault.*', re.IGNORECASE), 
                re.compile(r'.*assert.*', re.IGNORECASE), 
                re.compile(r'.*error.*', re.IGNORECASE)
            ]
        ) as reader:
            commander.reset_device()
            if is_unit_test:
                _start_ctrl_c_timer()
            
            stop_event = None
            if dump_audio_dir or dump_raw_spectrograms_dir or dump_quantized_spectrograms_dir:
                # Wait for the device to be ready
                while True:
                    reader.read(timeout=0.10)
                    # Check if any errors ocurred
                    if reader.error_message:
                        raise RuntimeError(f'Device error: {reader.error_message}')
                    if reader.started:
                        break

                stop_event = _start_jlink_processor(
                    dump_audio_dir=dump_audio_dir,
                    dump_raw_spectrograms_dir=dump_raw_spectrograms_dir,
                    dump_quantized_spectrograms_dir=dump_quantized_spectrograms_dir,
                    tflite_model_params=tflite_model_params
                )

            try:
                while not reader.read(timeout=.010):
                    time.sleep(0.005)

                if reader.error_message:
                    if stop_event is not None:
                        stop_event.set()
                    raise RuntimeError(f'Device error: {reader.error_message}')
            except KeyboardInterrupt:
                if stop_event is not None:
                    stop_event.set()


    ###############################################################
    def _run_audio_classifier_on_pc( 
        dump_audio_dir:str, 
        dump_raw_spectrograms_dir:str,
        dump_quantized_spectrograms_dir:str,
    ):
        """Run audio_classifier app on local PC"""
        nonlocal app_path

        if app_path is None:
            app_path = firmware_apps.get_image( 
                name='mltk_audio_classifier',
                accelerator=None,
                platform=platform,
                logger=logger
            )
        make_path_executable(app_path)

        tflite_name = tflite_model.filename or 'mltk_audio_classifier.tflite'
        tmp_tflite_path = create_tempdir('tmp_models') + f'/{os.path.splitext(tflite_name)[0]}.tflite'
        tflite_model.save(tmp_tflite_path)

        cmd = [app_path, '--model', tmp_tflite_path]
        if latency_ms is not None:
            cmd.extend(['--latency', str(latency_ms)])
        if dump_audio_dir:
            cmd.extend(['--dump_audio', dump_audio_dir])
        if dump_raw_spectrograms_dir:
            cmd.extend(['--dump_raw_spectrograms', dump_raw_spectrograms_dir])
        if dump_quantized_spectrograms_dir:
            cmd.extend(['--dump_spectrograms', dump_quantized_spectrograms_dir])

        env = os.environ
        if microphone:
            env['MLTK_MICROPHONE'] = microphone


        if is_unit_test:
            _start_ctrl_c_timer()

        cmd_str = ' '.join(cmd)
        logger.debug(cmd_str)
        retcode, retmsg = run_shell_cmd(cmd, outfile=logger, env=env)
        if retcode != 0:
            raise RuntimeError(f'Command failed (err code: {retcode}): {cmd_str}\n{retmsg}')


    ###############################################################
    def _generate_wav_from_dumped_audio(
        dump_dir:str, 
        sample_rate:int
    ):
        """Generate a .wav file from the dumped audio chunks generated by the audio_classifier app"""
        audio_data = bytearray()
        src_dir = f'{dump_dir}/bin'
        count = 0
        while True:
            p = f'{src_dir}/{count}.int16.bin'
            if not os.path.exists(p):
                break
            count += 1
            with open(p, 'rb') as f:
                audio_data.extend(f.read())

        if len(audio_data) == 0:
            return

        wav_path = f'{dump_dir}/dumped_audio.wav'
        with wave.open(wav_path,'w') as wav:
            # pylint: disable=no-member
            wav.setnchannels(1) # mono
            wav.setsampwidth(2) # 16-bit
            wav.setframerate(sample_rate)
            wav.writeframesraw(bytes(audio_data))

        logger.info(f'Generated: {wav_path}')


    ###############################################################
    def _dtype_to_str(dtype:np.dtype) -> str:
        if dtype in (np.int8, 'int8'):
            return 'int8'
        if dtype in (np.uint16, 'uint16'):
            return 'uint16'
        if dtype in (np.float32, 'float32'):
            return 'float32'
        raise RuntimeError(f'Unsupported dtype {dtype}')


    ###############################################################
    def _generate_video_from_dumped_spectrograms(
        dump_dir:str,
        dtype:np.dtype
    ):
        """Combine the genated .jpg spectrograms into an .mp4 video file"""
        spec_1_path = f'{dump_dir}/jpg/1.jpg'
        video_path = f'{dump_dir}/dump_spectrograms.mp4'

        dtype_str = _dtype_to_str(dtype)
        fps_name = f'{dtype_str}_spectrogram_fps'
        if fps_name not in globals():
            return
        spectrogram_fps = globals()[fps_name]

        
        if not os.path.exists(spec_1_path):
            return 
        spectrogram = cv2.imread(spec_1_path)
        height, width = spectrogram.shape[:2]

        logger.info(f'Spectrogram rate: {spectrogram_fps:.1f} frame/s')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, spectrogram_fps, (width, height)) 
        for i in range(1, int(1e9)):
            jpg_path = f'{dump_dir}/jpg/{i}.jpg'
            if not os.path.exists(jpg_path):
                break
            video.write(cv2.imread(jpg_path))
        video.release()
        logger.info(f'Generated: {video_path}')


    ###############################################################
    def _start_spectrogram_jpg_generator(
        dump_dir:str,
        dtype:np.dtype
    ):
        """Start a thread to periodically sample the spectrogram dump directory and generate a .jpg when one if found
        
        This converts from a numpy .txt array to a .jpg of the spectrogram
        """
        stop_event = threading.Event()

        dtype_str = _dtype_to_str(dtype) 
        src_dir = f'{dump_dir}/bin'
        dst_dir = f'{dump_dir}/jpg'
        os.makedirs(dst_dir, exist_ok=True)
        resize_dim_range = (240, 480) # Ensure the jpg dims are within this range

        def _convert_npy_to_jpg_loop():
            fps_list = collections.deque(maxlen=15)
            prev_time = None
            counter = 1
            while not stop_event.is_set():
                src_path = f'{src_dir}/{counter}.{dtype_str}.npy.txt'
                next_path = f'{src_dir}/{counter+1}.{dtype_str}.npy.txt'
                dst_path = f'{dst_dir}/{counter}.jpg'

                # We wait until the NEXT spectrogram file is found
                # this way, we avoid race-conditions and ensure the current 
                # spectrogram is fully written
                if not os.path.exists(next_path):
                    time.sleep(0.050)
                    continue 

                if prev_time is None:
                    prev_time = time.time()
                else:
                    now = time.time()
                    elapsed = (now - prev_time) or .1
                    prev_time = now
                    fps_list.append(elapsed)
                    globals()[f'{dtype_str}_spectrogram_fps'] = len(fps_list) / sum(fps_list)

                counter += 1

                try:
                    spectrogram = np.loadtxt(src_path, delimiter=',')
                except Exception as e:
                    logger.debug(f'Failed to read {src_path}, err: {e}')
                    continue

                # Transpose to put the time on the x-axis
                spectrogram = np.transpose(spectrogram)
                # Convert from int8 to uint8 
                if dtype_str == 'int8':
                    spectrogram = np.clip(spectrogram +128, 0, 255)
                    spectrogram = spectrogram.astype(np.uint8)
                elif dtype_str == 'uint16':
                    spectrogram = np.clip(spectrogram / 4, 0, 255)
                    spectrogram = spectrogram.astype(np.uint8)
                elif dtype_str == 'float32':
                    spectrogram = np.clip(spectrogram *255, 0, 255)
                    spectrogram = spectrogram.astype(np.uint8)

                jpg_data = cv2.applyColorMap(spectrogram, cv2.COLORMAP_HOT)
                jpg_data = _resize_jpg_image(jpg_data, *resize_dim_range)
                cv2.imwrite(dst_path, jpg_data)

        atexit.register(stop_event.set)
        t = threading.Thread(
            target=_convert_npy_to_jpg_loop, 
            name='Spectrogram to JPG converter',
            daemon=True,
        )
        t.start()


    ###############################################################
    def _start_jlink_processor(
        dump_audio_dir:str, 
        dump_raw_spectrograms_dir:str, 
        dump_quantized_spectrograms_dir:str,
        tflite_model_params:TfliteModelParameters
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
                dump_audio_dir=dump_audio_dir,
                dump_raw_spectrograms_dir=dump_raw_spectrograms_dir,
                dump_quantized_spectrograms_dir=dump_quantized_spectrograms_dir,
                tflite_model_params=tflite_model_params,
                logger=jlink_logger,
            )
        )
        t.start()
        return stop_event


    ###############################################################
    def _jlink_processing_loop(
        jlink_stream:JlinkStream, 
        stop_event:threading.Event,
        dump_audio_dir:str, 
        dump_raw_spectrograms_dir:str, 
        dump_quantized_spectrograms_dir:str, 
        tflite_model_params:TfliteModelParameters,
        logger:logging.Logger
    ):
        """Read binary data from embedded device via JLink interface
        
        This runs in a separate thread
        """
        audio_stream:JLinkDataStream = None 
        raw_spectrogram_stream:JLinkDataStream = None
        quantized_spectrogram_stream:JLinkDataStream = None
        audio_chunk_counter = 0
        spec_counter = 0
        #sample_rate = tflite_model_params['fe.sample_rate_hz']
        spectrogram_rows = tflite_model_params['fe.filterbank_n_channels']
        sample_length_ms = tflite_model_params['fe.sample_length_ms']
        window_size_ms = tflite_model_params['fe.window_size_ms']
        window_step_ms = tflite_model_params['fe.window_step_ms']
        spectrogram_cols = (sample_length_ms - window_size_ms) // window_step_ms + 1
        spectrogram_size = spectrogram_rows*spectrogram_cols
        raw_spectrogram_min_read_size = spectrogram_size*2
        quantized_spectrogram_min_read_size = spectrogram_size
        audio_min_read_size = 1024
        audio_start_timeout = None

        while True:
            if stop_event.wait(0.010):
                jlink_stream.disconnect()
                break 

            if dump_audio_dir and audio_stream is None:
                try:
                    audio_stream = jlink_stream.open('audio', mode='r')
                    logger.debug('Device audio stream ready')
                    # Wait a moment for any noise to be flushed from the audio stream
                    audio_start_timeout = time.time() + 4.0 
                except Exception as e:
                    logger.debug(f'Failed to open device audio stream, err: {e}')

            if dump_raw_spectrograms_dir and raw_spectrogram_stream is None:
                os.makedirs(dump_raw_spectrograms_dir, exist_ok=True)
                try:
                    raw_spectrogram_stream = jlink_stream.open('raw_spec', mode='r')
                    logger.debug('Device raw spectrogram stream ready')
                except Exception as e:
                    logger.debug(f'Failed to open device raw spectrogram stream, err: {e}')

            if dump_quantized_spectrograms_dir and quantized_spectrogram_stream is None:
                os.makedirs(dump_quantized_spectrograms_dir, exist_ok=True)
                try:
                    quantized_spectrogram_stream = jlink_stream.open('quant_spec', mode='r')
                    logger.debug('Device quantized spectrogram stream ready')
                except Exception as e:
                    logger.debug(f'Failed to open device quantized spectrogram stream, err: {e}')


            if audio_stream is not None and audio_stream.read_data_available >= audio_min_read_size:
                chunk_data = audio_stream.read_all(audio_stream.read_data_available)
                if time.time() - audio_start_timeout >= 0:
                    if audio_chunk_counter == 0:
                        logger.info('Audio recording started')
                    chunk_path = f'{dump_audio_dir}/{audio_chunk_counter}.int16.bin'
                    audio_chunk_counter += 1
                    with open(chunk_path, 'wb') as f:
                        f.write(chunk_data)

            if raw_spectrogram_stream is not None and raw_spectrogram_stream.read_data_available >= raw_spectrogram_min_read_size:
                spectrogram_read_size = (raw_spectrogram_stream.read_data_available // raw_spectrogram_min_read_size) * raw_spectrogram_min_read_size
                spectrogram_data = raw_spectrogram_stream.read_all(spectrogram_read_size)
                if spec_counter == 0:
                    logger.info('Raw spectrogram recording started')
                while len(spectrogram_data) > 0:
                    spectrogram_buf = np.frombuffer(spectrogram_data[:raw_spectrogram_min_read_size], dtype=np.uint16)
                    spectrogram_data = spectrogram_data[raw_spectrogram_min_read_size:]
                    spectrogram = np.reshape(spectrogram_buf, (spectrogram_cols, spectrogram_rows))
                    bin_path = f'{dump_raw_spectrograms_dir}/{spec_counter}.uint16.npy.txt'
                    np.savetxt(bin_path, spectrogram, fmt='%d', delimiter=',')
                    spec_counter += 1

            if quantized_spectrogram_stream is not None and quantized_spectrogram_stream.read_data_available >= quantized_spectrogram_min_read_size:
                spectrogram_read_size = (quantized_spectrogram_stream.read_data_available // quantized_spectrogram_min_read_size) * quantized_spectrogram_min_read_size
                spectrogram_data = quantized_spectrogram_stream.read_all(spectrogram_read_size)
                if spec_counter == 0:
                    logger.info('Quantized spectrogram recording started')
                while len(spectrogram_data) > 0:
                    spectrogram_buf = np.frombuffer(spectrogram_data[:quantized_spectrogram_min_read_size], dtype=input_dtype)
                    spectrogram_data = spectrogram_data[quantized_spectrogram_min_read_size:]
                    spectrogram = np.reshape(spectrogram_buf, (spectrogram_cols, spectrogram_rows))
                    bin_path = f'{dump_quantized_spectrograms_dir}/{spec_counter}.{_dtype_to_str(input_dtype)}.npy.txt'
                    np.savetxt(bin_path, spectrogram, fmt='%d', delimiter=',')
                    spec_counter += 1


    ###############################################################
    def _update_model_parameters() -> TfliteModelParameters: 
        """Update the .tflite embedded model parameters based on the command-line options"""
        params = TfliteModelParameters.load_from_tflite_model(tflite_model)

        if not (verbose or \
            average_window_duration_ms is not None or \
            detection_threshold or \
            suppression_ms is not None or \
            minimum_count or \
            volume_gain or  \
            dump_audio or  \
            dump_raw_spectrograms or \
            dump_quantized_spectrograms or \
            sensitivity or \
            latency_ms is not None):
            return params

        if verbose:
            params['verbose_model_output_logs'] = True
        if average_window_duration_ms:
            params['average_window_duration_ms'] = average_window_duration_ms
        if detection_threshold:
            params['detection_threshold'] = detection_threshold
        if suppression_ms:
            params['suppression_ms'] = suppression_ms
        if minimum_count:
            params['minimum_count'] = minimum_count
        if volume_gain:
            params['volume_gain'] = volume_gain
        if dump_audio:
            params['dump_audio'] = dump_audio
        if dump_raw_spectrograms:
            params['dump_raw_spectrograms'] = dump_raw_spectrograms
        if dump_quantized_spectrograms:
            params['dump_spectrograms'] = dump_quantized_spectrograms
        if sensitivity:
            params['sensitivity'] = sensitivity
        if latency_ms is not None:
            params['latency_ms'] = latency_ms

        params.add_to_tflite_model(tflite_model)
        return params


    ###############################################################
    def _resize_jpg_image(im:np.ndarray, min_dim:int, max_dim:int) -> np.ndarray:
        """Resize a jpg image based on the given dimensions"""
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


    ###############################################################
    def _start_ctrl_c_timer():
        """This is used for a unit test to simulate issuing CTRL+C """
        def _on_timeout():
            logger = cli.get_logger()
            logger.warning('Issuing CTRL+C\n')
            send_signal(signal.SIGINT, pid=0)
        t = threading.Timer(5, _on_timeout)
        t.setDaemon(True)
        t.setName('Ctrl+C Timer')
        t.start()



    ##################################################################################
    #
    # Actual command logic
    #


    tflite_model_params = _update_model_parameters()
    dump_audio_dir = None 
    dump_audio_bin_dir = None
    dump_raw_spectrograms_dir = None 
    dump_raw_spectrograms_bin_dir = None
    dump_quantized_spectrograms_dir = None 
    dump_quantized_spectrograms_bin_dir = None

    if dump_audio:
        sample_rate = tflite_model_params['fe.sample_rate_hz']
        dump_audio_dir = create_user_dir(f'audio_classifier_recordings/{platform}/audio')
        dump_audio_bin_dir = create_user_dir(f'audio_classifier_recordings/{platform}/audio/bin')
        logger.info(f'Dumping audio to {dump_audio_dir}')
        clean_directory(dump_audio_dir)
        atexit.register(functools.partial(
            _generate_wav_from_dumped_audio, 
            dump_dir=dump_audio_dir,
            sample_rate=sample_rate
        ))
    
    if dump_raw_spectrograms:
        dump_raw_spectrograms_dir = create_user_dir(f'audio_classifier_recordings/{platform}/raw_spectrograms')
        dump_raw_spectrograms_bin_dir = create_user_dir(f'audio_classifier_recordings/{platform}/raw_spectrograms/bin')
        logger.info(f'Dumping spectrograms to {dump_raw_spectrograms_dir}')
        clean_directory(dump_raw_spectrograms_dir)
        atexit.register(functools.partial(
            _generate_video_from_dumped_spectrograms, 
            dump_dir=dump_raw_spectrograms_dir,
            dtype='uint16'
        ))
        _start_spectrogram_jpg_generator(dump_raw_spectrograms_dir, 'uint16')

    if dump_quantized_spectrograms:
        dump_quantized_spectrograms_dir = create_user_dir(f'audio_classifier_recordings/{platform}/quantized_spectrograms')
        dump_quantized_spectrograms_bin_dir = create_user_dir(f'audio_classifier_recordings/{platform}/quantized_spectrograms/bin')
        logger.info(f'Dumping spectrograms to {dump_quantized_spectrograms_dir}')
        clean_directory(dump_quantized_spectrograms_dir)
        atexit.register(functools.partial(
            _generate_video_from_dumped_spectrograms, 
            dump_dir=dump_quantized_spectrograms_dir,
            dtype=input_dtype
        ))
        _start_spectrogram_jpg_generator(dump_quantized_spectrograms_dir, input_dtype)


    if use_device:
        _run_audio_classifier_on_device( 
            tflite_model_params=tflite_model_params,
            dump_audio_dir=dump_audio_bin_dir,
            dump_raw_spectrograms_dir=dump_raw_spectrograms_bin_dir,
            dump_quantized_spectrograms_dir=dump_quantized_spectrograms_bin_dir
        )
    else:
        _run_audio_classifier_on_pc(
            dump_audio_dir=dump_audio_bin_dir,
            dump_raw_spectrograms_dir=dump_raw_spectrograms_bin_dir,
            dump_quantized_spectrograms_dir=dump_quantized_spectrograms_bin_dir
        )


    