
from __future__ import annotations
import logging
import glob
import os
import re
import time
import queue
import threading
from multiprocessing.pool import ThreadPool
from typing import Tuple, Union, Callable, List
from concurrent.futures import ThreadPoolExecutor
from mltk.utils.python import (
    prepend_exception_msg
)
from mltk.utils.path import create_tempdir, fullpath
from mltk.utils.signal_handler import SignalHandler
from ..system import is_windows


try:
    import paramiko
except Exception:
    raise RuntimeError('Failed import paramiko Python package, try running: pip install paramiko OR pip install silabs-mltk[full]')


logging.raiseExceptions = False 


class SshClient(paramiko.client.SSHClient):
    """SSH client helper
    
    This extends the SSHClient object:
    https://docs.paramiko.org/en/stable/api/client.html
    """
    def __init__(
        self,
        logger:logging.Logger=None,
        environment:dict=None,
        compress:bool=False,
        sudo_password:str=None,
        bufsize:int=65536,
        shell:str='auto',
        remote_dir:str='.',
        connection_settings:dict=None
    ):
        super().__init__()

        self.logger = logger
        self.compress = compress
        self.bufsize = bufsize
        self.remote_dir = remote_dir
        self.shell = shell
        self.is_windows = False

        self._transport:paramiko.Transport = None
        self._environment= environment or {}
        self._sudo_password = sudo_password
        self._connection_settings = connection_settings
    
        self.set_missing_host_key_policy(paramiko.AutoAddPolicy())


    def connect(self, **kwargs): # pylint: disable=arguments-differ
        """Connect to the SSH server"""
        if not kwargs:
            kwargs = self._connection_settings

        hostname = kwargs.get('hostname', None)
        port = kwargs.get('port', None)
        key_filename = kwargs.get('key_filename', None)
        if key_filename:
            kwargs['key_filename'] = fullpath(key_filename)

        connect_msg = 'remote SSH server'
        if hostname:
            connect_msg = hostname
        if port:
            connect_msg += f':{port}'

        try:
            self.logger.info(f'Connecting to {connect_msg}')
            super().connect(**kwargs)

            transport = self.get_transport()
            transport.use_compression(self.compress)
            self.logger.debug('Connected')
        except Exception as e:
            prepend_exception_msg(e, f'Failed to connect to {connect_msg}')
            raise


        if self.shell == 'auto':
            self.shell = None
            self.is_windows = True
            self.logger.debug('Determining remote OS')
            retcode, _ = self.execute_command('ver.exe', log_level=1, raise_exception_on_error=False)
            if retcode == 0:
                self.logger.info('Remote OS is Windows')
                self.is_windows = True
            else:
                self.logger.info('Remote OS is Unix')
                self.is_windows = False 
                self.shell = 'sh'

        if self.is_windows:
            self.python_exe = 'python'
        else:
            self.python_exe = 'python3'


    def close(self):
        """Close the connection, ignoring any errors"""
        try:
            super().close()
        except:
            pass


    def __enter__(self):
        if self.get_transport() is None:
            self.connect()
        return self

    def __exit__(self ,type_, value, traceback):
        self.close()


    def execute_command(
        self,
        cmd:str,
        timeout:float=None,
        input_data:str=None,
        raise_exception_on_error=True,
        background:bool=False,
        log_level=logging.INFO,
        log_line_parser:Callable[[str],None]=None,
        handle_ctrl_c=True,
        maxlen=4096,
    ) -> Union[Tuple[int, str], SshBackgroundCommand]:
        """Execute a shell command on the SSH server machine"""
        if input_data is None and cmd.startswith('sudo'):
            input_data = self._sudo_password

        if self.shell:
            if self.shell == 'cmd':
                cmd = f'{self.shell} /c {cmd}'
            else:
                cmd = f'{self.shell} -c \'{cmd}\''

        try:
            if background:
                if self.is_windows:
                    raise RuntimeError('Background commands not currently supported with Windows')
                
                handle = SshBackgroundCommand(
                    func=self._run_poll,
                    cmd=cmd,
                    timeout=timeout, 
                    log_level=log_level, 
                    input_data=input_data,
                    log_line_parser=log_line_parser,
                    maxlen=maxlen,
                    handle_ctrl_c=False
                )

                return handle

            else:
                retcode, retmsg = self._run_poll(
                    cmd=cmd,
                    timeout=timeout, 
                    input_data=input_data,
                    log_level=log_level,
                    log_line_parser=log_line_parser,
                    handle_ctrl_c=handle_ctrl_c,
                    maxlen=maxlen
                )
    
                if retcode != 0 and raise_exception_on_error:
                    raise RuntimeError(f'Command failed: {cmd}, err: {retmsg}')

                return retcode, retmsg

        except paramiko.SSHException as e:
            prepend_exception_msg(e, f'Failed to execute command: {cmd}')
            raise


    def execute_batch_commands(
        self,
        cmds:List[str],
        timeout:float=None,
        input_data:str=None,
        raise_exception_on_error=True,
        log_level=logging.INFO,
        log_line_parser:Callable[[str],None]=None,
        handle_ctrl_c=True,
        maxlen=4096
    ) -> Tuple[int, str]:
        """Execute a list of shell commands on the SSH server machine"""
        if self.is_windows:
            raise RuntimeError('Batch commands not currently supported with Windows')

        if input_data is None and any(e.startswith('sudo') for e in cmds):
            input_data = self._sudo_password


        local_batch_file = f'{create_tempdir()}/mltk_batch_cmds.sh'
        remote_batch_file = f'{self.remote_dir}/.mltk_batch_cmds-{int(time.time())}.sh'

        for i,(k,v) in enumerate(self._environment.items()):
            cmds.insert(i, f'export {k}="{v}"')
        cmds.append(f'rm -f {remote_batch_file}')

        self.logger.debug('-' * 80)
        self.logger.log(log_level, 'Batch script that will execute on remote server:')
        with open(local_batch_file, 'w', newline='\n', encoding='utf-8') as f:
            for cmd in cmds:
                f.write(cmd + '\n')
                self.logger.log(log_level, cmd)
        self.logger.debug('-' * 80)

        self.upload_file(local_batch_file, remote_batch_file)

        try:
            cmd = f'{self.shell} {remote_batch_file}'
            retcode, retmsg = self._run_poll(
                cmd=cmd,
                timeout=timeout, 
                input_data=input_data,
                log_level=log_level, 
                log_line_parser=log_line_parser,
                handle_ctrl_c=handle_ctrl_c,
                maxlen=maxlen
            )
    
            if retcode != 0 and raise_exception_on_error:
                raise RuntimeError(f'Batch commands failed, err: {retmsg}')

            return retcode, retmsg

        except paramiko.SSHException as e:
            prepend_exception_msg(e, 'Failed to execute batch commands')
            raise
        finally:
            os.remove(local_batch_file)



    def upload_file(
        self, 
        local_path:str, 
        remote_path:str=None
    ):
        """Upload a file to the SSH server machine"""
        sftp_client = self.open_sftp()
        local_path = fullpath(local_path)
        remote_path = remote_path or f'{self.remote_dir}/{os.path.basename(local_path)}'
        
        try:
            remote_dir = os.path.dirname(remote_path).replace('\\', '/')
            if remote_dir:
                self.create_remote_dir(remote_dir)
            self.logger.debug(f'Uploading: {local_path} to: {remote_path}')
            self._change_remote_cwd(sftp_client, self.remote_dir, remote_path)
            sftp_client.put(local_path, remote_path)
        finally:
            try:
                sftp_client.close()
            except:
                pass


    def upload_files(
        self, 
        paths:List[str], 
        cwd:str=None,
        remote_cwd:str=None
    ):
        """Upload multiple files to the SSH server machine"""
        orig_cwd = os.getcwd()
        cwd = cwd or orig_cwd
        remote_cwd = remote_cwd or self.remote_dir
        resolved_paths = []

        for path in paths:
            if '|' in path:
                parts = path.split('|')

                if '*' in path:
                    raise RuntimeError(f'Invalid upload path: {path}\nPaths with a pipe "|" must not contain wild cards')
                if len(parts) != 2:
                    raise RuntimeError(f'Invalid upload path: {path}\nIf a pipe "|" is in the path then it should have the format: <local path>|<remote path>')
                
                local_path = fullpath(parts[0], cwd=cwd)
                remote_path = parts[1]

                if not os.path.exists(local_path):
                    raise FileNotFoundError(f'Invalid upload path: {path}\nLocal path not found: {local_path}')
                
                resolved_paths.append((local_path, remote_path))
            else:
                try:
                    os.chdir(cwd)
                    glob_files = glob.glob(path, recursive=True)
                finally:
                    os.chdir(orig_cwd)

                for local_path in glob_files:
                    local_path = cwd + '/' + local_path.replace('\\', '/')
                    if not os.path.isfile(local_path):
                        continue
                    relpath = os.path.relpath(local_path, cwd).replace('\\', '/')
                    remote_path = f'{remote_cwd}/{relpath}'.replace('\\', '/')
                    resolved_paths.append((local_path, remote_path))

        sftp_client = self.open_sftp()

        try:
            sftp_client.chdir(remote_cwd)
            created_remote_dirs = []
            for (local_path, remote_path) in resolved_paths:
                remote_dir = os.path.dirname(remote_path).replace('\\', '/')
                if remote_dir and remote_dir not in created_remote_dirs:
                    created_remote_dirs.append(remote_dir)
                    self.create_remote_dir(remote_dir, remote_cwd=remote_cwd)
                
                self.logger.debug(f'Uploading: {local_path} to: {remote_path}')
                self._change_remote_cwd(sftp_client, remote_cwd, remote_path)
                sftp_client.put(local_path, remote_path)
        finally:
            try:
                sftp_client.close()
            except:
                pass


    def download_file(
        self, 
        remote_path:str, 
        local_path:str=None,
        raise_exception=True
    ) -> str:
        """Download a file from the SSH server"""
        sftp_client = self.open_sftp()
        sftp_client.sock.timeout = 30

        if not local_path:
            local_path = create_tempdir('temp/ssh/downloads') + f'/{os.path.basename(remote_path)}'
        local_path = fullpath(local_path)

        resolved_remote_path = self.resolve_remote_path(remote_path)
        if not resolved_remote_path:
            if raise_exception:
                raise FileNotFoundError(f'File not found on remote: {remote_path}')
            else:
                return None

        try:
            self.logger.debug(f'Downloading: {resolved_remote_path} to: {local_path}')
            local_dir = os.path.dirname(local_path)
            if local_dir:
                os.makedirs(local_dir, exist_ok=True)
            sftp_client.get(resolved_remote_path, local_path)
        except Exception as e:
            try:
                os.remove(local_path)
            except:
                pass
            if raise_exception:
                raise
            self.logger.debug(f'Failed to download {resolved_remote_path}, err: {e}')
        finally:
            try:
                sftp_client.close()
            except:
                pass

        return local_path


    def download_files(
        self,
        paths:List[str], 
        cwd:str=None,
        remote_cwd:str=None,
        raise_exception=True,
        parallel_count:int=1
    ) -> List[str]:
        """Download multiple files from the SSH server"""
        cwd = cwd or os.getcwd()
        remote_cwd = remote_cwd or self.remote_dir
        resolved_paths = []

        for path in paths:
            if '|' in path:
                parts = path.split('|')
                if '*' in path:
                    raise RuntimeError(f'Invalid download path: {path}\nPaths with a pipe "|" must not contain wild cards')
                if len(parts) != 2:
                    raise RuntimeError(f'Invalid download path: {path}\nIf a pipe "|" is in the path then it should have the format: <remote path>|<local path>')

                remote_path = self.resolve_remote_path(parts[0], cwd=remote_cwd, raise_exception=raise_exception)
                if not remote_path:
                    continue
                
                local_path = fullpath(parts[1], cwd=cwd)
                resolved_paths.append((remote_path, local_path))

            else:
                try:
                    if '*' in path:
                        resolved_remote_paths = self.resolve_glob_remote_paths(
                            path=path,
                            cwd=remote_cwd
                        )
                    else:
                        resolved_remote_path = self.resolve_remote_path(
                            path, 
                            cwd=remote_cwd, 
                            raise_exception=raise_exception
                        )
                        if not resolved_remote_path:
                            continue
                        resolved_remote_paths = [resolved_remote_path]

                    for remote_path in resolved_remote_paths:
                        relpath = os.path.relpath(remote_path, remote_cwd).replace('\\', '/')
                        local_path = f'{cwd}/{relpath}'
                        resolved_paths.append((remote_path, local_path))

                except Exception as e:
                    if raise_exception:
                        raise
                    self.logger.debug(f'Failed to resolve remote paths, err: {e}')
                    return []

        if not resolved_paths:
            return []

        pool = ThreadPool(processes=parallel_count)
        parallel_count = min(parallel_count, len(resolved_paths))
        resolved_path_chunks = (
            resolved_paths[i:i+parallel_count] \
            for i in range(0, len(resolved_paths), parallel_count)
        )

        downloaded_local_paths = []
        sftp_clients = [None] * parallel_count
        cancelled_download = threading.Event()
        exception_msg = []

        def _cancel_download():
            self.logger.info('Cancelling download')
            cancelled_download.set()
            _close_clients()
            if raise_exception:
                return 'forward-signal'

        def _close_clients():
            for client in sftp_clients:
                try:
                    if client:
                        client.close()
                except:
                    pass
    
        try:
            results = []
            client_index = 0
            for path_chunk in resolved_path_chunks:
                client_index = (client_index + 1) % len(sftp_clients)
                results.append(
                    pool.apply_async(self._download_files, 
                    kwds=dict(
                        client_index=client_index,
                        clients=sftp_clients,
                        paths=path_chunk,
                        cancelled_download=cancelled_download
                )))
            
           
            with SignalHandler(callback=_cancel_download):
                while len(results) > 0:
                    modified_list = False
                    for result in results:
                        if result.ready():
                            result_value = result.get()
                            if isinstance(result_value, list):
                                downloaded_local_paths.extend(result_value)
                            else:
                                exception_msg.append(result_value)

                            results.remove(result) # pylint: disable=modified-iterating-list
                            modified_list = True
                            break

                    if not modified_list:
                        time.sleep(0.010)

            if exception_msg and raise_exception:
                raise RuntimeError('\n'.join(exception_msg))
                
        finally:
            self.logger.debug('Closing file download thread pool')
            _close_clients()
            try:
                pool.close()
            except:
                pass

        self.logger.debug(f'Downloaded {len(downloaded_local_paths)} of {len(resolved_paths)} files')
            
        return downloaded_local_paths


    def _download_files(
        self, 
        client_index:int, 
        clients:List[paramiko.SFTPClient], 
        paths:List[Tuple[str]],
        cancelled_download:threading.Event,
    ) -> Union[List[str], str]:
        """Internal helper function to download files from the SSH server"""
        downloaded_local_paths = []
        for (remote_path, local_path) in paths:
            self.logger.debug(f'Downloading: {remote_path} to: {local_path}')
            local_dir = os.path.dirname(local_path)
            if local_dir:
                os.makedirs(local_dir, exist_ok=True)

            for retries in range(1, 4):
                if cancelled_download.is_set():
                    return downloaded_local_paths

                try:
                    if clients[client_index] is None:
                        clients[client_index] = self.open_sftp()
                        clients[client_index].sock.timeout = 15

                    clients[client_index].get(remote_path, local_path)
                    downloaded_local_paths.append(local_path)
                    break
                except Exception as e:
                    try:
                        clients[client_index].close()
                    except:
                        pass 
                    clients[client_index] = None

                    try:
                        os.remove(local_path)
                    except:
                        pass

                    if not cancelled_download.is_set():
                        self.logger.debug(f'Attempt {retries} of 3: Failed to download remote file: {remote_path}, err: {e}')

                    if retries < 3 and 'No such file' not in f'{e}':
                        time.sleep(0.500)
                        continue 
                    return f'{e}'

        return downloaded_local_paths

    
    def create_remote_dir(
        self,
        remote_dir:str,
        remote_cwd:str=None,
    ):
        """Create a directory on the SSH server"""
        remote_cwd = remote_cwd or self.remote_dir
        self.logger.debug(f'Creating directory on remote: {remote_dir}')
        if self.is_windows:
            remote_cwd = self._normalize_remote_path(remote_cwd)
            remote_dir = self._normalize_remote_path(remote_dir)
            mkdir_cmd = f'mkdir "{remote_dir}" 2> NUL'
        else:
            mkdir_cmd = f'mkdir -p {remote_dir}'
        
        self.execute_command(
            f'cd "{remote_cwd}" && {mkdir_cmd}', 
            log_level=1, 
            raise_exception_on_error=False
        )


    def resolve_glob_remote_paths(self, path:str, cwd:str=None) -> List[str]:
        """List files on the SSH server using glob"""
        cwd = cwd or self.remote_dir
        cwd = self._normalize_remote_path(cwd)

        cmd = f'cd "{cwd}" && {self.python_exe} -c "import glob,os; r=glob.glob(\\"{path}\\", recursive=True); print(\\"MLTK_START_PATHS=\\" + \\";\\".join(p for p in r if os.path.isfile(p)));"'
        _, retmsg = self.execute_command(cmd, log_level=1, maxlen=-1)
        idx = retmsg.index('MLTK_START_PATHS=')
        if idx == -1:
            return []
        retmsg = retmsg[idx + len('MLTK_START_PATHS='):]
        return retmsg.strip().split(';')


    def resolve_remote_path(self, path:str, cwd:str=None, raise_exception=True) -> str:
        """Resolve a file path on the SSH server"""
        cwd = cwd or self.remote_dir
        cwd = self._normalize_remote_path(cwd)

        cmd = f'cd "{cwd}" && {self.python_exe} -c "from os.path import expandvars,expanduser,normpath,abspath; print(\\"MLTK_REMOTE_PATH=\\"+abspath(normpath(expanduser(expandvars(\\"{path}\\")))))"'
        retcode, retmsg = self.execute_command(cmd, log_level=1, maxlen=-1, raise_exception_on_error=False)
        if retcode == 0:
            idx = retmsg.index('MLTK_REMOTE_PATH=')
            retmsg = retmsg[idx + len('MLTK_REMOTE_PATH='):]
            return retmsg.strip()

        if raise_exception:
            raise FileNotFoundError(f'File not found on remote: {path}')
        self.logger.debug(f'File not found on remote: {path}')

        return None


    def kill_process(self, pid:int):
        """Kill a process by ID"""
        if self.is_windows:
            self.execute_command(f'taskkill /F /PID {pid}', raise_exception_on_error=False, log_level=1) 
        else:
            self.execute_command(f'pkill -P {pid}', raise_exception_on_error=False, log_level=1) 


    def _normalize_remote_path(self, remote_path:str) -> str:
        if self.is_windows:
            return remote_path.replace('/', '\\')
        return remote_path


    def _change_remote_cwd(self, sftp_client:paramiko.SFTPClient, remote_cwd:str, remote_path:str):
        # Only change the remote directory if the remote is non-Windows OR we were not given a Windows absolute path.
        # For Windows, we chdir(None) due to an incompatibility in paramiko
        remote_path = self._normalize_remote_path(remote_path)
        remote_cwd = remote_cwd if not (self.is_windows and re.match(r'[a-z]\:\\.*', remote_path, flags=re.IGNORECASE)) else None
        sftp_client.chdir(remote_cwd)


    def _run_poll(
        self, 
        cmd:str,
        timeout:float, 
        input_data:str,
        log_level:int,
        log_line_parser:Callable[[str],None],
        handle_ctrl_c:bool,
        maxlen:int,
        cancel_event:threading.Event=None,
    ):
        if log_level >= logging.DEBUG:
            self.logger.debug(cmd)
        
        session = self._transport.open_session()
        session.set_combine_stderr(True)
        if not self.is_windows:
            session.get_pty(width=0, height=0, term='dumb')
        session.update_environment(self._environment)

        stdin = session.makefile_stdin("wb", self.bufsize)
        stdout = session.makefile("r", self.bufsize)

        session.exec_command(cmd)

        try:
            return self._run_poll_unsafe(
                session=session,
                stdin=stdin,
                stdout=stdout, 
                log_level=log_level,
                timeout=timeout,
                input_data=input_data,
                cancel_event=cancel_event,
                log_line_parser=log_line_parser,
                handle_ctrl_c=handle_ctrl_c,
                maxlen=maxlen
            )
        finally:
            try:
                stdin.close()
            except:
                pass
            try:
                stdout.close()
            except:
                pass
            try:
                session.close()
            except:
                pass


    def _run_poll_unsafe(
        self, 
        session:paramiko.Channel,
        stdin:paramiko.ChannelStdinFile,
        stdout:paramiko.ChannelFile,
        log_level:int, 
        timeout:float, 
        input_data:str,
        cancel_event:threading.Event,
        log_line_parser:Callable[[str],None],
        handle_ctrl_c:bool,
        maxlen:int
    ):

        if hasattr(self.logger, 'flush'):
            flush_func = self.logger.flush 

        retcode = 0
        line_buf = ''
        out_buf = ''

        def _write_line(data:str):
            nonlocal line_buf, out_buf
            line_buf += data
            out_buf += data
            if maxlen >= 0 and len(out_buf) > maxlen:
                out_buf = out_buf[len(out_buf)-maxlen:]


            crlf_index = line_buf.find('\r\n')
            lf_index = line_buf.find('\n')
            if crlf_index != -1:
                line =  line_buf[:crlf_index]
                line_buf = line_buf[crlf_index+2:]
            elif lf_index != -1:
                line =  line_buf[:lf_index]
                line_buf = line_buf[lf_index+1:]
                return
            else:
                return

            if log_line_parser is not None:
                log_line_parser(line)

            try:
                self.logger.log(log_level, line)
            except:
                return 
            
            if flush_func is not None:
                try:
                    flush_func()
                except:
                    pass

        def _enqueue_output(file, q):
            for line in iter(file.readline, ''):
                q.put(line)
            file.close()

        def _on_ctrl_c():
            session.close()
            return 'forward-signal'
        

        def _rx_loop():
            nonlocal retcode, input_data

            with ThreadPoolExecutor(1) as pool:
                q_stdout = queue.Queue()
                pool.submit(_enqueue_output, stdout, q_stdout)
                start_time = time.time()

                while True:
                    if (session.exit_status_ready() or session.closed) and q_stdout.empty():
                        break

                    if not q_stdout.empty():
                        _write_line(q_stdout.get_nowait())

                    if input_data is not None and session.send_ready():
                        stdin.write(input_data + '\n')
                        stdin.flush()
                        input_data = None

                    if timeout is not None and (time.time() - start_time) > timeout:
                        retcode = -1
                        break

                    if cancel_event is not None and cancel_event.is_set():
                        retcode = -1
                        break

                    time.sleep(0.001)

        if handle_ctrl_c and threading.current_thread() is threading.main_thread():
            with SignalHandler(callback=_on_ctrl_c):
                _rx_loop()
        else:
            _rx_loop()

        if retcode == 0:
            retcode = session.recv_exit_status()

        return retcode, out_buf






class SshBackgroundCommand:
    def __init__(
        self,
        func,
        **kwargs
    ):
        self.retcode = 0
        self.retmsg = ''
        self._cancel_event = threading.Event()
        self._func = func
        kwargs['cancel_event'] = self._cancel_event
        self._thread = threading.Thread(
            target=self._run,
            name='background cmd',
            kwargs=kwargs
        ) 
        self._thread.setDaemon(True)
        self._thread.start()

    def cancel(self):
        self._cancel_event.set() 


    def wait(self) -> Tuple[int, str]:
        self._thread.join()
        return self.retcode, self.retmsg

    def _run(self, **kwargs):
        self.retcode, self.retmsg = self._func(**kwargs)