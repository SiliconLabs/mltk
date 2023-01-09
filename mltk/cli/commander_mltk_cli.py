import os
import sys
import re
import time
import logging
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


    logger = cli.get_logger()
    try:
        issue_command(*ctx.meta['vargs'], outfile=logger)
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
    firmware_image_path:str = typer.Argument(..., help='Path to firmware executable'),
    platform:str = typer.Option(None, help='Platform name'),
    masserase:bool = typer.Option(False, help='Mass erase device before programming firmware image'),
    device:str = typer.Option(None, help='JLink device code'),
    setup_script:str = typer.Option(None, help='Path to python script to execute before programing device'),
    setup_script_args:str = typer.Option(None, help='Arguments to pass to setup script'),
    port:str = typer.Option(None, help='Serial COM port'),
    baud:int = typer.Option(None, help='Serial COM port BAUD'),
    timeout:float = typer.Option(60, help='Maximum time in seconds to wait for program to complete on device'),
    host:str = typer.Option(None, help='SSH host name if this should execute remotely'),
    verbose:bool = typer.Option(False, '-v', '--verbose', help='Enable verbose logging'),
    completed_msg:str = typer.Option(None, help='Regex for app to print to console for it to have successfully completed'),
    retries:int = typer.Option(0, help='The number of times to retry running the firmware on the device'),
):
    """Run a firmware image on a device and parse its serial output for errors"""
    from mltk.utils.path import fullpath, create_tempdir
    from mltk.utils.shell_cmd import run_shell_cmd
    from mltk.utils import commander
    from mltk.utils.commander.commander import masserse_device
    from mltk.utils.serial_reader import SerialReader

    logger = cli.get_logger(verbose=verbose)

    prev_pid_path = create_tempdir('tmp') + '/build_download_run_prev_pid.txt'
    logger.error(f'PID={os.getpid()}')

    if setup_script:
        setup_script = fullpath(setup_script)
        if not os.path.exists(setup_script):
            cli.abort(msg=f'Invalid argument, --setup-script, file not found: {setup_script}')


    if host:
        try:
            _download_run_on_remote(
                host=host,
                firmware_image_path=firmware_image_path,
                platform=platform,
                device=device,
                setup_script=setup_script,
                setup_script_args=setup_script_args,
                port=port,
                baud=baud,
                masserase=masserase,
                completed_msg=completed_msg,
                timeout=timeout,
                verbose=verbose,
                logger=logger,
                prev_pid_path=prev_pid_path,
                retries=retries
            )
        except Exception as e:
            cli.handle_exception('Failed to run command on remote', e)
        return

    stop_regex =[re.compile(r'.*done.*', re.IGNORECASE)]
    if completed_msg:
        logger.debug(f'Completed msg regex: {completed_msg}')
        stop_regex.append(re.compile(completed_msg, re.IGNORECASE))

    firmware_image_path = fullpath(firmware_image_path)

    if setup_script:
        cmd = f'{sys.executable} "{setup_script}" {setup_script_args if setup_script_args else ""}'
        logger.info(f'Running setup script: {cmd}')
        retcode, _ = run_shell_cmd(cmd, logger=logger, outfile=logger)
        if retcode != 0:
            cli.abort(retcode, 'Failed to execute setup script')

    if masserase:
        masserse_device(platform=platform, device=device)

    logger.info(f'Programming {firmware_image_path} to device ...')
    commander.program_flash(
        firmware_image_path,
        platform=platform,
        device=device,
        show_progress=False,
        halt=True,
    )

    # If no serial COM port is provided,
    # then attemp to resolve it based on common Silab's board COM port description
    port = port or 'regex:JLink CDC UART Port'
    baud = baud or 115200

    max_retries = max(retries, 1)
    for retry_count in range(1, max_retries+1):
        logger.error(f'Executing application on device (attempt {retry_count} of {max_retries}) ...')
        logger.debug(f'Opening serial connection, BAUD={baud}, port={port}')
        with SerialReader(
            port=port,
            baud=baud,
            outfile=logger,
            stop_regex=stop_regex,
            fail_regex=[
                re.compile(r'.*hardfault.*', re.IGNORECASE),
                re.compile(r'.*error.*', re.IGNORECASE),
                re.compile(r'.*failed to alloc memory.*', re.IGNORECASE),
                re.compile(r'.*assert failed.*', re.IGNORECASE)
            ]
        ) as serial_reader:
            # Reset the board to start the profiling firmware
            commander.reset_device(platform=platform, device=device, logger=logger)

            # Wait for up to a minute for the profiler to complete
            # The read() will return when the stop_regex, fail_regex, or timeout condition is met
            if not serial_reader.read(timeout=timeout):
                logger.error('Timed-out waiting for app on device to complete')
                if retry_count < max_retries:
                    serial_reader.close()
                    time.sleep(3.0) # Wait a moment and retry
                    continue

                cli.abort()

            # Check if the profiler failed
            if serial_reader.failed:
                logger.error(f'App failed on device, err: {serial_reader.error_message}')
                if retry_count < max_retries:
                    serial_reader.close()
                    time.sleep(3.0) # Wait a moment and retry
                    continue

                cli.abort()

            break

    logger.info('Application successfully executed')



def _download_run_on_remote(
    host:str,
    firmware_image_path:str,
    platform:str,
    device:str,
    masserase:bool,
    setup_script:str,
    setup_script_args:str,
    port:str,
    baud:int,
    timeout:float,
    verbose:bool,
    completed_msg:str,
    logger:logging.Logger,
    prev_pid_path:str,
    retries:int
):

    from mltk.utils.ssh import SshClient
    from mltk.utils.path import fullpath
    from mltk.utils.system import get_username
    from mltk.utils.commander import commander as command_module
    import paramiko

    ssh_config_path = fullpath('~/.ssh/config')
    if not os.path.exists(ssh_config_path):
        raise FileNotFoundError(f'SSH config not found: {ssh_config_path}')
    ssh_config_obj = paramiko.SSHConfig.from_path(ssh_config_path)

    ssh_config = ssh_config_obj.lookup(host)
    if 'identityfile' not in ssh_config:
        raise ValueError(f'{ssh_config_path} must contain the "host" entry: {host}, with an "IdentityFile" value')

    connection_settings = {}
    connection_settings['hostname'] = ssh_config['hostname']
    connection_settings['key_filename'] = fullpath(ssh_config['identityfile'][0])
    if 'user' in ssh_config:
        connection_settings['username'] = ssh_config['user']
    if 'port' in ssh_config:
        connection_settings['port'] = ssh_config['port']


    with SshClient(logger=logger, connection_settings=connection_settings) as ssh_client:
        if os.path.exists(prev_pid_path):
            try:
                with open(prev_pid_path, 'r') as f:
                    prev_pid = f.read().strip()
                    logger.info(f'Killing previous process: {prev_pid}')
                    ssh_client.kill_process(pid=prev_pid)
            finally:
                os.remove(prev_pid_path)

        retcode, retmsg = ssh_client.execute_command(f'{ssh_client.python_exe} -c "import tempfile;print(f\\"MLTK_REMOTE_PATH=\\"+tempfile.gettempdir())"')
        if retcode != 0:
            raise RuntimeError(f'Failed to get tmpdir on remote, err: {retmsg}')
        idx = retmsg.index('MLTK_REMOTE_PATH=')
        remote_tmp_dir = retmsg[idx + len('MLTK_REMOTE_PATH='):].strip().replace('\\', '/')
        ssh_client.remote_dir = f'{remote_tmp_dir}/{get_username()}/mltk/remote_mltk'
        logger.info(f'Creating MLTK venv at: {ssh_client.remote_dir}')

        ssh_client.create_remote_dir(ssh_client.remote_dir, remote_cwd='.')


        retcode, retmsg = ssh_client.execute_command(f'{ssh_client.python_exe} -m venv {ssh_client.remote_dir}')
        if retcode != 0:
            raise RuntimeError(f'Failed to create MLTK venv, err: {retmsg}')

        if ssh_client.is_windows:
            pip_exe = f'{ssh_client.remote_dir}/Scripts/pip'.replace('/', '\\')
            python_exe = f'{ssh_client.remote_dir}/Scripts/python'.replace('/', '\\')
            mltk_exe = f'{ssh_client.remote_dir}/Scripts/mltk'.replace('/', '\\')
        else:
            pip_exe = f'{ssh_client.remote_dir}/bin/pip3'
            python_exe = f'{ssh_client.remote_dir}/bin/python3'.replace('/', '\\')
            mltk_exe = f'{ssh_client.remote_dir}/bin/mltk'.replace('/', '\\')

        retcode, retmsg = ssh_client.execute_command(f'{pip_exe} install silabs-mltk --upgrade')
        if retcode != 0:
            raise RuntimeError(f'Failed to install MLTK into remote venv, err: {retmsg}')

        retcode, retmsg = ssh_client.execute_command(f'{python_exe} -c "import os;import mltk;print(f\\"MLTK_REMOTE_PATH=\\"+os.path.dirname(mltk.__file__))"')
        if retcode != 0:
            raise RuntimeError(f'Failed to get mltk path on remote, err: {retmsg}')
        idx = retmsg.index('MLTK_REMOTE_PATH=')
        remote_mltk_dir = retmsg[idx + len('MLTK_REMOTE_PATH='):].strip().replace('\\', '/')

        firmware_image_path = fullpath(firmware_image_path)
        remote_firmwage_image_path = f'{ssh_client.remote_dir}/{os.path.basename(firmware_image_path)}'

        ssh_client.upload_file(__file__, f'{remote_mltk_dir}/cli/command_mltk_cli.py')
        ssh_client.upload_file(command_module.__file__, f'{remote_mltk_dir}/utils/commander/commander.py')
        ssh_client.upload_file(firmware_image_path, remote_firmwage_image_path)

        cmd = f'{mltk_exe} build download_run {remote_firmwage_image_path}'

        if setup_script:
            remote_setup_script = f'{ssh_client.remote_dir}/{os.path.basename(setup_script)}'
            ssh_client.upload_file(setup_script, remote_setup_script)
            cmd += f' --setup-script "{remote_setup_script}"'
            if setup_script_args:
                cmd += f' --setup-script-args "{setup_script_args}"'

        if platform:
            cmd  += f' --platform {platform}'
        if device:
            cmd  += f' --device {device}'
        if port:
            cmd  += f' --port "{port}"'
        if baud:
            cmd  += f' --baud {baud}'
        if timeout:
            cmd  += f' --timeout {timeout}'
        if completed_msg:
            cmd += f' --completed-msg "{completed_msg}"'
        if verbose:
            cmd += ' --verbose'
        if masserase:
            cmd += ' --masserase'
        if retries:
            cmd += f' --retries {retries}'


        pid_re = re.compile(r'.*PID=(\d+).*')
        def _log_line_parser(line:str):
            match = pid_re.match(line)
            if match:
                with open(prev_pid_path, 'w') as f:
                    f.write(match.group(1))

        logger.debug(f'Executing on remote: {cmd}')
        retcode, retmsg = ssh_client.execute_command(cmd, log_line_parser=_log_line_parser)
        if retcode != 0:
            raise RuntimeError(f'Failed to execute MLTK command on remote: {cmd.join(" ")}')