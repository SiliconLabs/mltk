
import re
import typer
from mltk import cli



@cli.root_cli.command("commander", cls=cli.VariableArgumentParsingCommand)
def silabs_commander_command(ctx: typer.Context):
    """Silab's Commander Utility

    This utility allows for accessing a Silab's embedded device via JLink.

    For more details issue command: mltk commander --help
    """

    # Import all required packages here instead of at top
    # to help improve the CLI's responsiveness
    from mltk.utils.commander import issue_command

    help_re = re.compile(r'Usage:\s.*\[command\]\s\[options\].*')
    def _line_parser(l):
        if help_re.match(l):
            return 'Usage: mltk commander [command] [options]\n'
        return l


    logger = cli.get_logger()
    try:
        issue_command(*ctx.meta['vargs'], outfile=logger, line_processor=_line_parser)
    except Exception as e:
        cli.handle_exception('Commander failed', e)



@cli.root_cli.command('program_app')
def program_app_command(
    firmware_image_path:str = typer.Argument(..., help='Path to firmware executable'),
    model: str = typer.Option(None, '--model', '-m',
        help='''\bOne of the following:
- Name of previously trained MLTK model
- Path to .tflite model file
- Path to .mltk.zip model archive file''',
        metavar='<model>'
    ),
    platform:str = typer.Option(None, help='Platform name. If omitted then platform is automatically determined based on the connected device'),
    verbose:bool = typer.Option(False, '-v', '--verbose', help='Enable verbose logging'),
):
    """Program the given firmware image to the connected device"""
    from mltk.utils import firmware_apps
    from mltk.utils.path import fullpath
    from mltk.core import load_tflite_model

    tflite_model = None
    logger = cli.get_logger(verbose=verbose)

    if model:
        try:
            tflite_model = load_tflite_model(
                model,
                print_not_found_err=True,
                logger=logger
            )
        except Exception as e:
            cli.handle_exception('Failed to load model', e)

    app_name = None
    accelerator = None

    if re.match(r'.*\..*', firmware_image_path):
        firmware_image_path = fullpath(firmware_image_path)
    else:
        toks = firmware_image_path.split('-')
        firmware_image_path = None
        if len(toks) >= 3:
            accelerator = toks[-1]
            platform = toks[-2]
        if len(toks) == 3:
            app_name = toks[0]
        elif len(toks) == 4:
            app_name = '-'.join(toks[:2])
        else:
            cli.abort(msg='Invalid firmware image path argument')


    firmware_apps.program_image_with_model(
        name=app_name,
        platform=platform,
        accelerator=accelerator,
        tflite_model=tflite_model,
        logger=logger,
        halt=False,
        firmware_image_path=firmware_image_path,
    )





@cli.build_cli.command('download_run', hidden=True)
def download_run_command(
    firmware_image_path:str = typer.Argument(..., help='Path to firmware executable to program to device'),
    platform:str = typer.Option(None, help='Platform name'),
    masserase:bool = typer.Option(False, help='Mass erase device before programming firmware image'),
    device:str = typer.Option(None, help='JLink device code'),
    serial_number:str = typer.Option(None, help='J-Link debugger USB serial number'),
    ip_address:str = typer.Option(None, help='J-Link debugger IP address'),
    setup_script:str = typer.Option(None, help='Path to python script to execute before programming device'),
    setup_script_args:str = typer.Option(None, help='Arguments to pass to setup script'),
    program_script:str = typer.Option(None, help='Path to python script to execute to program device. If omitted, then use Commander to program device'),
    program_script_args:str = typer.Option(None, help='Arguments to pass to program script. If the value _IMAGE_PATH_ is in the args string, then it will automatically be replaced with the given firmware_image_path'),
    reset_script:str = typer.Option(None, help='Path to python script to reset device after programming. If omitted, then use Commander to reset the device'),
    reset_script_args:str = typer.Option(None, help='Arguments to pass to reset script'),
    port:str = typer.Option(None, help='Serial COM port'),
    baud:int = typer.Option(None, help='Serial COM port BAUD'),
    timeout:float = typer.Option(60, help='Maximum time in seconds to wait for program to complete on device'),
    verbose:bool = typer.Option(False, '-v', '--verbose', help='Enable verbose logging'),
    start_msg:str = typer.Option(None, help='Regex for app to print to console for it the serial logger to start recording'),
    completed_msg:str = typer.Option(None, help='Regex for app to print to console for it to have successfully completed'),
    retries:int = typer.Option(0, help='The number of times to retry running the firmware on the device'),
    remote_address:str = typer.Option(None, help='Address of remote execution server'),
    log_path:str = typer.Option(None, help='Log file path'),
    dump_log_file:bool = typer.Option(False, help='Dump the log file on error'),
):
    """Run a firmware image on a device and parse its serial output for errors"""
    from mltk.utils.commander import download_image_and_run
    from mltk.utils.commander.download_run_server import download_image_and_run as download_image_and_run_on_remote
    from mltk.utils.logger import get_logger

    logger = get_logger(
        'downloader_run', 
        level='DEBUG', 
        console=True, 
        log_file=log_path
    ) if log_path else cli.get_logger(verbose=verbose)

    logger.verbose = verbose

    try:
        if remote_address:
            download_image_and_run_on_remote(
                firmware_image_path=firmware_image_path,
                platform=platform,
                masserase=masserase,
                device=device,
                serial_number=serial_number,
                ip_address=ip_address,
                setup_script=setup_script,
                setup_script_args=setup_script_args,
                program_script=program_script,
                program_script_args=program_script_args,
                reset_script=reset_script,
                reset_script_args=reset_script_args,
                port=port,
                baud=baud,
                timeout=timeout,
                start_msg=start_msg,
                completed_msg=completed_msg,
                retries=retries,
                remote_address=remote_address,
                logger=logger
            )
        else:
            download_image_and_run(
                firmware_image_path=firmware_image_path,
                platform=platform,
                masserase=masserase,
                device=device,
                serial_number=serial_number,
                ip_address=ip_address,
                setup_script=setup_script,
                setup_script_args=setup_script_args,
                program_script=program_script,
                program_script_args=program_script_args,
                reset_script=reset_script,
                reset_script_args=reset_script_args,
                port=port,
                baud=baud,
                timeout=timeout,
                start_msg=start_msg,
                completed_msg=completed_msg,
                retries=retries,
                logger=logger
            )
    except Exception as e:
        logger.exception('Failed to download image and run on device', exc_info=e)
        if dump_log_file:
            _dump_log_file(logger)
        cli.abort(msg=f'{e}')
   

def _dump_log_file(logger):
    try:
        with open(logger.file_handler_path, 'r') as f:
            for line in f:
                cli.print_error(line.rstrip())
    except:
        pass

@cli.root_cli.command('start_download_run_server', hidden=True)
def start_download_run_server_command(
    address:str =  typer.Option('0.0.0.0:50051', help='Listening address of server'),
    verbose:bool = typer.Option(False, '-v', '--verbose', help='Enable verbose logging'),
    log_path:str = typer.Option(None, help='Log file path'),
):
    """Start the download run server
    
    This is used by the "mltk build download_run_remote" command
    """
    from mltk.utils.commander.download_run_server import DownloadRunServer
    from mltk.utils.logger import get_logger

    logger = get_logger(
        'downloader_run', 
        level='DEBUG', 
        console=verbose, 
        log_file=log_path
    ) if log_path else cli.get_logger(verbose=verbose)
    
    _server = DownloadRunServer(
        logger=logger,
        address=address
    )

    try:
        _server.start()
    except KeyboardInterrupt:
        _server.stop()


