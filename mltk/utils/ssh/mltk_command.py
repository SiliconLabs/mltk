from __future__ import annotations
from copy import deepcopy
import re
import logging
import os
import time
import json
from dataclasses import dataclass
import pprint
from typing import Union, List, Tuple

import paramiko

import mltk
from mltk.core import MltkModel, load_mltk_model
from mltk.utils.path import get_user_setting, create_user_dir, fullpath, remove_directory, create_tempdir
from mltk.utils.python import prepend_exception_msg
from mltk.utils.logger import DummyLogger
from mltk.utils.signal_handler import SignalHandler
from mltk.utils.archive import gzip_directory_files

from .ssh_client import SshClient


def get_settings() -> dict:
    """Return the SSH settings from ~/.mtlk/user_settings.yaml"""
    return get_user_setting('ssh', dict())

def get_setting(name:str, default=None):
    """Return a specific SSH setting from ~/.mtlk/user_settings.yaml"""
    settings:dict = get_settings()
    return settings.get(name, default)


def run_mltk_command(
    cmd:List[str],
    ssh_host:str=None,
    ssh_port:int=None,
    ssh_key_path:str=None,
    ssh_password:str=None,
    connection_settings:dict=None,
    clean:bool=False,
    force:bool=False,
    resume_only:bool=False,
    wait_for_results:bool=True,
    local_dir:str=None,
    remote_dir:str=None,
    local_log_dir:str=None,
    environment:Union[list,dict]=None,
    upload_files:list=None,
    download_files:list=None,
    startup_cmds:list=None,
    shutdown_cmds:list=None,
    create_venv:bool=None,
    logger:logging.Logger=None,
):
    """Run an MLTK command on a remote machine via SSH"""
    if len(cmd) < 2:
        raise ValueError('MLTK cmd must contain at least 2 arguments, e.g.: train my_model')
    
    # Find the MLTK Model file
    try:
        mltk_model = load_mltk_model(
            cmd[1],  
            test=cmd.count('--test') > 0,
            print_not_found_err=True
        )
    except Exception as e:
        prepend_exception_msg(e, 'Failed to load model')
        raise 
    # Update the command's "model" argument to be the model's name
    # (in case it was the file path to its python script)
    cmd[1] = mltk_model.name
    if mltk_model.test_mode_enabled:
        cmd[1] += '-test'
    
    logger = logger or DummyLogger()
    remote_dir = remote_dir or _get_ssh_model_setting('remote_dir', mltk_model)
    upload_files = upload_files or _get_ssh_model_setting('upload_files', mltk_model) or []
    download_files = download_files or _get_ssh_model_setting('download_files', mltk_model)
    environment = environment or _get_ssh_model_setting('environment', mltk_model)
    startup_cmds = startup_cmds or _get_ssh_model_setting('startup_cmds', mltk_model)
    shutdown_cmds = shutdown_cmds or _get_ssh_model_setting('shutdown_cmds', mltk_model)
    create_venv = create_venv if create_venv is not None else _get_ssh_model_setting('create_venv', mltk_model, default=get_setting('create_venv', True))

    upload_files.append(os.path.basename(mltk_model.model_specification_path))
    local_dir = local_dir or os.path.dirname(mltk_model.model_specification_path)
    local_log_dir = local_log_dir or  mltk_model.log_dir


    # Open an SSH connection to the remote server
    with _open_connection(
        ssh_host=ssh_host,
        ssh_port=ssh_port,
        ssh_key_path=ssh_key_path,
        ssh_password=ssh_password,
        connection_settings=connection_settings,
        remote_dir=remote_dir,
        environment=environment,
        logger=logger,
    ) as ssh_client:
        # Check if an existing MLTK command is active
        proc_context = _get_previous_remote_process(
            ssh_client=ssh_client,
            logger=logger
        )

        # If not command was previously executed
        if not proc_context:
            if resume_only:
                raise RuntimeError('Cannot resume, no command was previously executed')

        # If the MLTK command is still active on the remote server
        elif proc_context.is_active:
            # If we only want to resume the previous command then just return
            if resume_only:
                logger.info(f'Remote MLTK command: {proc_context} is still active on the remote server')

            # Otherwise, if resume_only=false (i.e. we want to start a new command)
            else:
                # If a previous MLTK command is active and force=true
                # then kill the previous command before starting the new one
                if force:
                    _kill_previous_remote_process(
                        ssh_client=ssh_client,
                        logger=logger
                    )

                # Else raise an error so that only one MLTK command is active on the remote server
                else:
                    raise RuntimeError(
                        f'An existing MLTK command is active on the remote server: {proc_context}\n' \
                        'Hint: Add the --force option to kill the previous command before starting a new one.'
                    )

        # Else if a previous MLTK command was executed but is no longer active
        else:
            # If we only want to resume the previous command,
            # then finalize the previous command now and return
            if resume_only:
                _finalize_command(
                    ssh_client=ssh_client,
                    proc_context=proc_context,
                    activate_venv=create_venv,
                    download_files=download_files,
                    shutdown_cmds=shutdown_cmds,
                    local_dir=local_dir,
                    local_log_dir=local_log_dir,
                    logger=logger
                )
                return

            # Otherwise, ensure the user provided the --force option
            # so we do not accidentally discard the previous command's results
            elif force:
                logger.warning(f'Discarding previous remote cmd: {proc_context}')
                proc_context.clear_cache()
            
            else:
                raise RuntimeError(
                    f'An existing MLTK command has previously completed on the remote server: {proc_context}\n' \
                    'Use the --resume option to finalize the previous command\n' \
                    'OR use the --force option to discard the previous command'
                )
               

        # If we're not only resuming a previous command
        # then start the new command
        if not resume_only:
            proc_context = _start_command(
                cmd=cmd,
                ssh_client=ssh_client,
                mltk_model=mltk_model,
                local_dir=local_dir,
                local_log_dir=local_log_dir,
                upload_files=upload_files,
                startup_cmds=startup_cmds,
                create_venv=create_venv,
                clean=clean,
                logger=logger,
            )

        # If we don't want to wait for the command to complete
        # then just return
        if not wait_for_results:
            logger.info(f'To retrieve the command results, issue the command:\nmltk ssh {" ".join(cmd)} --resume')
            return

        try:
            # Wait for the command to complete on the remote server
            _wait_for_command(
                ssh_client=ssh_client,
                proc_context=proc_context,
                logger=logger,
            )
        finally:
            # Finalize the remotely executed command
            # by copying any remote files to the local machine
            _finalize_command(
                ssh_client=ssh_client,
                proc_context=proc_context,
                activate_venv=create_venv,
                download_files=download_files,
                shutdown_cmds=shutdown_cmds,
                local_log_dir=local_log_dir,
                local_dir=local_dir,
                logger=logger
            )


def _open_connection(
    ssh_host:str,
    ssh_port:int,
    ssh_key_path:str,
    ssh_password:str,
    connection_settings:dict,
    remote_dir:str,
    environment:Union[list,dict],
    logger:logging.Logger,
) -> SshClient:
    """Open an SSH connection"""
    try:
        ssh_config = None
        ssh_config_path = fullpath(get_setting('config_path', '~/.ssh/config'))
        if os.path.exists(ssh_config_path):
            logger.debug(f'SSH config path: {ssh_config_path}')
            ssh_config = paramiko.SSHConfig.from_path(ssh_config_path)
    except Exception as e:
        logger.warning(f'Failed to parse SSH config file: {ssh_config_path}, err: {e}')

    remote_dir = remote_dir or get_setting('remote_dir', '.')

    if not connection_settings:
        connection_settings = get_setting('connection', {})

    host, port, user, path = _parse_ssh_host(ssh_host)

    ssh_host_config = {}
    if ssh_config is not None and host:
        ssh_host_config = ssh_config.lookup(host)
        if len(ssh_host_config) <= 1:
            ssh_host_config = {}
        else:
            logger.debug(f'Found host:{host} in {ssh_config_path}\n{pprint.pformat(ssh_host_config)}')
        
    if 'hostname' in ssh_host_config:
        connection_settings['hostname'] = ssh_host_config['hostname']
    elif host:
        connection_settings['hostname'] = host

    if ssh_port:
        connection_settings['port'] = ssh_port
    elif port:
        connection_settings['port'] = port
    elif 'port' in ssh_host_config:
        connection_settings['port'] = ssh_host_config['port']


    if user:
        connection_settings['username'] = user
    elif 'user' in ssh_host_config:
        connection_settings['username'] = ssh_host_config['user']
    
    if ssh_key_path:
        connection_settings['key_filename'] = fullpath(ssh_key_path)
    elif 'identityfile' in ssh_host_config:
        connection_settings['key_filename'] = fullpath(ssh_host_config['identityfile'][0])


    if path:
        remote_dir = path

    if ssh_password:
        connection_settings['password'] = ssh_password
    
    s = deepcopy(connection_settings)
    if 'password' in s:
        s['password'] = '****'
    logger.debug(f'Connection settings:\n{pprint.pformat(s, indent=3)}')


    if not connection_settings:
        raise RuntimeError('Must specify connection_settings as argument or in ~/.mltk/user_settings.yaml')
    if 'hostname' not in connection_settings:
        raise RuntimeError('Must specify value SSH host')
        

    combined_env = _parse_environment(get_setting('environment', {}))
    combined_env.update(_parse_environment(environment))

    ssh_client = SshClient(
        logger=logger,
        environment=combined_env,
        sudo_password=connection_settings.get('password', None),
        shell='bash --login', # Execute commands in a login BASH shell
        remote_dir=remote_dir
    )
    ssh_client.connect(**connection_settings)

    return ssh_client


def _start_command(
    cmd:List[str],
    ssh_client:SshClient,
    mltk_model:MltkModel,
    local_dir:str,
    local_log_dir:str,
    upload_files:List[str],
    startup_cmds:List[str],
    create_venv:bool,
    clean:bool,
    logger:logging.Logger,
) -> ProcessContext:
    """Execute an MLTK command on a remote machine"""
    if clean and cmd[0] == 'train':
        cmd.append('--clean')
        logger.info(f'Cleaning {local_log_dir}')
        remove_directory(local_log_dir)
        try:
            os.remove(mltk_model.archive_path)
            logger.info(f'Cleaned {mltk_model.archive_path}')
        except:
            pass

    sync_local_mltk = get_setting('sync_local_mltk', False)
    combined_upload_files = get_setting('upload_files', [])
    combined_upload_files.extend(upload_files or [])

    combined_startup_cmds = get_setting('startup_cmds', [])
    combined_startup_cmds.extend(startup_cmds or [])
    remote_dir = ssh_client.remote_dir

    # If we want to sync to local MLTK with the remote, then:
    # 1) .tar.gz all the python files in the local MLTK
    # 2) Upload the .tar.gz archive to the remote
    # 3) After installing the MLTK on the remote, untar the archive into the remote MLTK directory
    if sync_local_mltk:
        logger.warning('Syncing local MLTK with remote')
        local_mltk_archive_path = gzip_directory_files(
            src_dir=mltk.MLTK_DIR,
            dst_archive=create_tempdir('tmp') + '/local_mltk.tar.gz',
            regex=r'.*\.py$'
        )
        logger.debug(f'Generated {local_mltk_archive_path} from {mltk.MLTK_DIR}')
        combined_upload_files.append(f'{local_mltk_archive_path}|.mltk_tmp/local_mltk.tar.gz')

    # Create to remote_dir and resolve the actual path (this includes any additional settings specified in ~/.bash_profile on remote)
    logger.info(f'Changing remote working directory to: {remote_dir}')
    _, retmsg = ssh_client.execute_command(f'mkdir -p "{remote_dir}" && cd "{remote_dir}" &&  pwd', log_level=logging.DEBUG)
    remote_dir = ssh_client.remote_dir = retmsg.splitlines()[-1].strip()
    logger.info(f'Full remote directory path: {remote_dir}')
    
    proc_context = ProcessContext(
        cmd=cmd,
        remote_dir=remote_dir
    )

    # Create the MLTK python venv (if necessary)
    if create_venv:
        logger.info('Creating MLTK python virtual environment on remote server')
        ssh_client.execute_command(f'cd "{remote_dir}" && python3 -m venv .venv', log_level=logging.DEBUG)

        logger.info('Installing the MLTK into the remote virtual environment')
        try:
            ssh_client.execute_command(f'cd "{remote_dir}" && . ./.venv/bin/activate && pip3 install wheel silabs-mltk=={mltk.__version__}', log_level=logging.DEBUG)
        except:
            # If we failed to install the current MLTK version, then try just installing the latest on pypi
            ssh_client.execute_command(f'cd "{remote_dir}" && . ./.venv/bin/activate && pip3 install --upgrade wheel silabs-mltk', log_level=logging.DEBUG)

    # Uploading any local files to the remote server
    logger.info('Copying files from local machine to remote server')
    ssh_client.upload_files(
        paths=combined_upload_files,
        cwd=local_dir,
        remote_cwd=remote_dir,
    )

    batch_cmds = _prepare_batch_cmds(
        cmds=combined_startup_cmds,
        remote_dir=remote_dir,
        activate_venv=create_venv
    )
    batch_cmds.append('which mltk') # This is useful for debugging

    if sync_local_mltk:
        batch_cmds.append('python3 -c "import os;import mltk;os.system(\'tar -xf .mltk_tmp/local_mltk.tar.gz -C \' + mltk.MLTK_DIR);"')

    mltk_cmd_str = 'mltk ' + ' '.join(cmd)
    batch_cmds.append(f'rm -rf "{proc_context.remote_cmd_log_file}"')

    print_log_dir_cmd = f'python3 -c "import os;os.environ[\\"CUDA_VISIBLE_DEVICES\\"]=\\"-1\\";from mltk.core import load_mltk_model;m=load_mltk_model(\'{proc_context.model_name}\');print(\'MLTK_LOG_DIR=\'+m.log_dir);"'
    print_pid_cmd = 'echo "SUBPROCESS_PID=$BASHPID"'

    # We want the MLTK command to run in a detached subprocess
    # This way, if the SSH session prematurely closes it will continue to run
    batch_cmds.append(f'(set -m;  {print_log_dir_cmd} && {print_pid_cmd} && {mltk_cmd_str} > {proc_context.remote_cmd_log_file} &)')
    batch_cmds.append('sleep 30') # wait a moment for the subprocess to start
   
        
    # The bash script will print its process id (pid) and MLTK script logging dir
    # Parse this info from the logger
    SUBPROCESS_PID_RE = re.compile(r'SUBPROCESS_PID=(\d+)')
    MLTK_LOG_DIR_RE = re.compile(r'MLTK_LOG_DIR=(.+)')
    def _parse_log_line(line:str):
        match = SUBPROCESS_PID_RE.match(line.strip())
        if match:
            proc_context.pid = match.group(1)
        match = MLTK_LOG_DIR_RE.match(line.strip())
        if match:
            proc_context.remote_log_dir = match.group(1)

    # Execute the bash script on the remote server
    logger.info('Invoking MLTK command on remote server\n(This may take awhile, please be patient ...)')
    ssh_client.execute_batch_commands(
        batch_cmds, 
        log_level=logging.DEBUG,
        log_line_parser=_parse_log_line
    )


    if not proc_context:
        raise RuntimeError('Failed to invoke MLTK command on remote server')

    logger.debug(f'Saving remote process context to {ProcessContext.cache_path()}')
    proc_context.save()

    return proc_context
     

def _wait_for_command(
    ssh_client:SshClient,
    proc_context:ProcessContext,
    logger:logging.Logger
):
    """Wait for an MLTK command executing on a remote machine to complete"""
    kill_background_subprocess = False
    remote_error_log_path = None
    remote_error_log_re = re.compile(r'For more details see: (.*)')
    ssh_client.remote_dir = proc_context.remote_dir

    def _log_line_parser(line:str):
        nonlocal remote_error_log_path
        match = remote_error_log_re.match(line.strip())
        if match:
            remote_error_log_path = match.group(1)

    logger.info('Waiting for remote command to complete ...')
    logger.info('NOTE: Press CTRL+C to cancel, remote files will still be copied to the local machine')


    # Periodically poll the mltk cmd log file and print its contents to the logger
    # This is done in a background thread
    background_cmd = ssh_client.execute_command(
        f'tail -f -n +1 {proc_context.remote_cmd_log_file}', 
        log_level=logging.INFO, 
        background=True,
        log_line_parser=_log_line_parser
    )

    # In the main thread, periodically check that the detached subprocess
    # is running on the remote server is still active
    try:
        # Check if the user issues a Ctrl+C in the local terminal
        # In this case, we also want to abort the detached subprocess running on the remote server
        def _on_ctrl_c():
            nonlocal kill_background_subprocess
            kill_background_subprocess = True
            return 'forward-signal'
        
        with SignalHandler(callback=_on_ctrl_c):
            while True:
                if remote_error_log_path:
                    logger.debug('Errors detected by MLTK command executing on remote server, aborting')
                    break 

                retcode, _ = ssh_client.execute_command(
                    f'ps -p {proc_context.pid} > /dev/null', 
                    log_level=1, 
                    raise_exception_on_error=False,
                    handle_ctrl_c=False
                )
                if retcode != 0:
                    time.sleep(2.0) # short delay to ensure the remote subprocess's logs are all streamed
                    break 
                time.sleep(1.0)
    
    finally:
        # Stop the background command that is polling the remote log file
        background_cmd.cancel()

        # If necessary, kill the detached subprocess running on the remote server
        # This is done when the user issue Ctrl+C on the local terminal
        if kill_background_subprocess:
            _kill_previous_remote_process(
                ssh_client=ssh_client,
                logger=logger
            )
        # If an log file was generated on the remote server
        # and we have the remote file path
        # then download the remote file and dump it to the local logger
        if remote_error_log_path:
            try:
                logger.debug('\n' + '-' * 80)
                logger.debug(f'Dumping remote log file: {remote_error_log_path}')
                local_error_log_path = ssh_client.download_file( 
                    remote_path=remote_error_log_path
                )
                with open(local_error_log_path, 'r') as f:
                    for line in f:
                        logger.debug(line.rstrip())
            except:
                pass
            finally:
                try:
                    os.remove(local_error_log_path)
                except:
                    pass


def _finalize_command(
    ssh_client:SshClient,
    proc_context:ProcessContext,
    activate_venv:bool,
    local_dir:str,
    local_log_dir:str,
    download_files:List[str],
    shutdown_cmds:List[str],
    logger:logging.Logger
):
    """Finalize a remote MLTK command by downloading a relavant files"""
    ssh_client.remote_dir = proc_context.remote_dir
    
    combined_download_files = get_setting('download_files', [])
    combined_download_files.extend(download_files or [])

    combined_shutdown_cmds = get_setting('shutdown_cmds', [])
    combined_shutdown_cmds.extend(shutdown_cmds or [])


    logger.debug('Attempting to download model archive')
    if ssh_client.download_files(
            paths=[proc_context.model_archive_name], 
            cwd=local_dir,
            raise_exception=False
        ):
        logger.info(f'Model archive downloaded to local path: {local_dir}/{proc_context.model_archive_name}')

    if combined_download_files:
        logger.info('Attempting to download additional files')
        ssh_client.download_files(
            paths=combined_download_files, 
            cwd=local_dir,
            raise_exception=False
        )

    logger.info(
        f'Downloading log files from remote server: {proc_context.remote_log_dir}\n' \
        f'to local directory: {local_log_dir}\n' \
        '(This may take awhile, please be patient, press CTRL+C to cancel download)')

    remote_cmd_log_file = proc_context.remote_cmd_log_file
    if remote_cmd_log_file:
        ssh_client.download_file(
            remote_path=remote_cmd_log_file, 
            local_path=f'{local_log_dir}/{os.path.basename(remote_cmd_log_file)}',
            raise_exception=False
        )

    remote_cli_log_file = proc_context.remote_cli_log_file
    if remote_cli_log_file:
        ssh_client.download_file(
            remote_path=remote_cli_log_file, 
            local_path=f'{local_log_dir}/{os.path.basename(remote_cli_log_file)}',
            raise_exception=False
        )

    ssh_client.download_files(
        paths=[f'{proc_context.remote_log_dir}/**'], 
        remote_cwd=proc_context.remote_log_dir,
        cwd=local_log_dir,
        raise_exception=False
    )


    if combined_shutdown_cmds:
        logger.info('Executing shutdown commands on remote server')
        batch_cmds = _prepare_batch_cmds(
            cmds=combined_shutdown_cmds,
            remote_dir=ssh_client.remote_dir,
            activate_venv=activate_venv,
            stop_error=False,
        )
        ssh_client.execute_batch_commands(
            cmds=batch_cmds,
            log_level=logging.DEBUG,
            raise_exception_on_error=False
        )

    proc_context.clear_cache()



def _kill_previous_remote_process(
    ssh_client:SshClient,
    logger:logging.Logger
):
    """Kill an MLTK command process executing on a remote machine"""
    ctx = ProcessContext.load_from_cache()

    try:
        if ctx:
            logger.info(f'Killing remote process: {ctx}')
            ssh_client.execute_command(
                f'pkill -P {ctx.pid}', 
                raise_exception_on_error=False, 
                log_level=logging.DEBUG
            )
    except Exception as e:
        logger.warning(f'Failed to kill remote process, err: {e}')

    finally:
        ctx.clear_cache()


def _get_previous_remote_process(
    ssh_client:SshClient,
    logger:logging.Logger
) -> ProcessContext:
    """When a remote command is started on a remote machine, it's context information is 
    cacehd to ~/.mltk/ssh_cmd_process. 
    This loads that information and checks if it's still active on the remote machine"""
    ctx = ProcessContext.load_from_cache()
    if ctx:
        logger.debug(f'Checking if remote process: {ctx} is still active on the remote server ...')
        retcode, _ = ssh_client.execute_command(
            f'ps -p {ctx.pid} > /dev/null', 
            log_level=1, 
            raise_exception_on_error=False,
            handle_ctrl_c=True
        )

        ctx.is_active = retcode == 0
        logger.debug(f'Remote process: {ctx} {"is" if ctx.is_active else "is not"} activate')

    return ctx


def _prepare_batch_cmds(
    cmds:List[str],
    remote_dir:str,
    activate_venv:bool,
    stop_error:bool=True
) -> str:
    """Prepare a BASH script that will execute on the remote machine
    """
    remote_dir_cmd = None 
    activate_venv_cmd = None

    if remote_dir != '.':
        remote_dir_cmd = f'cd {remote_dir}'
    
    if activate_venv is True:
        activate_venv_cmd = f'. {remote_dir}/.venv/bin/activate'

    # Populate the commands to run in a bash script on the remote server
    batch_cmds = [
        'set -x', # Echo individual batch cmds to aid debugging
    ]
    if stop_error:
        batch_cmds.append('set -e') # Stop if an individual batch cmd fails

    if remote_dir_cmd:
        batch_cmds.append(remote_dir_cmd)
    batch_cmds.append('pwd') # useful for debugging
    if activate_venv_cmd:
        batch_cmds.append(activate_venv_cmd)

    batch_cmds.extend(cmds)

    return batch_cmds


def _get_ssh_model_setting(key:str, mltk_model:MltkModel, default=None):
    """Retrieve a setting from an MltkModel instance that inherits the SshMixin"""
    value = default
    if mltk_model.attributes.contains(f'ssh.{key}'):
        value = getattr(mltk_model, f'ssh_{key}')
    return value


def _parse_environment(environment:Union[list,dict]):
    """Parse the environment variable setting
    This optional converts the settings from a list to a dictionary"""
    if isinstance(environment, dict):
        return environment
    elif not environment:
        return {}

    retval = {}

    for e in environment:
        idx = e.find('=')
        if idx == -1:
            retval[e] = ''
        else:
            key = e[:idx]
            value = e[idx+1:]
            if (value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'"):
                value = value[1:-1]
            retval[key] = value

    return retval


def _parse_ssh_host(ssh_host:str) -> Tuple[str, str, int, str]:
    """Parse the --host CLI argument"""
    if not ssh_host:
        return None, None, None, None

    match = re.match(r'^((?P<user>.*?)@)?(?P<host>.*?)(:(?P<port>.*?))?(/(?P<path>.*))?$', ssh_host)
    if not match:
        return None, None, None, None

    user = match.group('user')
    host = match.group('host')
    port = match.group('port')
    path = match.group('path')

    if port:
        port = int(port)

    return host, port, user, path


@dataclass
class ProcessContext:
    """This class holds the remote command's process information"""
    pid:str = ''
    cmd:List[str] = None
    remote_dir:str = ''
    remote_log_dir:str = ''
    is_active:bool = False


    @property
    def cmd_str(self) -> str:
        if not self.cmd:
            return 'null'
        return ' '.join(self.cmd)

    @property
    def mltk_cmd(self) -> str:
        if not self.cmd:
            return ''
        return self.cmd[0] 

    @property
    def model_name(self) -> str:
        if not self.cmd:
            return ''
        return self.cmd[1]

    @property
    def model_archive_name(self) -> str:
        if not self.cmd:
            return ''
        return f'{self.model_name}.mltk.zip'

    @property
    def remote_cmd_log_file(self) -> str:
        return f'{self.remote_dir}/mltk_ssh_{self.mltk_cmd}_{self.model_name}.log'

    @property
    def remote_cli_log_file(self) -> str:
        if not self.remote_log_dir:
            return None
        base_log_dir = os.path.dirname(os.path.dirname(self.remote_log_dir)).replace('\\', '/')
        return f'{base_log_dir}/cli_logs/{self.mltk_cmd}.log'


    def __str__(self) -> str:
        return f'[{self.cmd_str}] (pid:{self.pid})'

    def __bool__(self) -> bool:
        return self.pid != ''

    def save(self):
        with open(ProcessContext.cache_path(), 'w') as f:
            json.dump(dict(
                pid=self.pid,
                cmd=self.cmd,
                remote_dir=self.remote_dir,
                remote_log_dir=self.remote_log_dir
            ), f, indent=3)


    def clear_cache(self):
        try:
            os.remove(ProcessContext.cache_path())
        except:
            pass


    @staticmethod
    def cache_path() -> str:
        return create_user_dir() + '/ssh_cmd_process.json' 

    
    @staticmethod
    def load_from_cache() -> ProcessContext:
        retval = ProcessContext()
        try:
            with open(ProcessContext.cache_path(), 'r') as f:
                data =  json.load(f)
                retval = ProcessContext(**data)
        except:
            retval = ProcessContext() 

        return retval




