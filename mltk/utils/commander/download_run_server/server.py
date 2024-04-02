from typing import Iterable
import sys
import os
import logging
import threading 
import queue
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor


try:
    import grpc 
except ModuleNotFoundError:
    print('Must install "grpc" python package')
    print('pip install grpc')
    sys.exit(-1)

from mltk.utils.logger import get_logger
from mltk.utils.commander import download_image_and_run
from mltk.utils.archive import extract_archive
from mltk.utils.path import remove_directory


_curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_curdir, 'proto'))
if not __package__:                  
    sys.path.insert(0, os.path.dirname(_curdir))
    __package__ = os.path.basename(_curdir) # pylint: disable=redefined-builtin


from .proto .download_run_pb2 import (Request, Response, Status) # pylint: disable=no-name-in-module
from .proto.download_run_pb2_grpc import (
  DownloadRunServicer,
  add_DownloadRunServicer_to_server
)

class DownloadRunServer(DownloadRunServicer):
    def __init__(
        self,
        address:str='localhost:50051',
        logger:logging.Logger=None
    ):
        super().__init__()

        self._lock = threading.Semaphore()
        self._address = address
        self._logger = logger or get_logger()
        self._server: grpc.Server = None 


    def start(self):
        self._server = grpc.server(
            ThreadPoolExecutor(),
            compression=grpc.Compression.Deflate,
            options=[
                ('grpc.max_connection_idle_ms', 60*1000),
                ('grpc.client_idle_timeout_ms', 60*1000),
                ('grpc.keepalive_time_ms', 15*1000),
                ('grpc.keepalive_timeout_ms', 30*1000)
            ]
        )
        add_DownloadRunServicer_to_server(self, self._server)
        self._server.add_insecure_port(self._address)
        self._logger.info(f'Starting DownloadRun server on {self._address}')
        self._server.start()
        self._server.wait_for_termination()


    def stop(self):
        try:
            self._logger.info('Stopping server')
            self._server.stop(5.0)
        except:
            pass


    def DownloadAndRun(
        self, 
        request:Request, # pylint: disable=no-member
        context:grpc.RpcContext
    ) -> Iterable[Response]: # pylint: disable=no-member
        self._logger.info(f'Acquiring lock (timeout={request.lock_timeout}) ...')
        if not self._lock.acquire(timeout=request.lock_timeout):
            self._logger.warning('Cannot handle request, the server is currently busy')
            yield Response(
                status=Status.Timeout,
                message='The server is processing another request'
            )
        else:
            try:
                self._logger.info('#' * 80)
                req_msg = f'Processing request:\n{_request_to_string(request)}'
                self._logger.info(req_msg)
                yield Response(
                    status=Status.DebugLog,
                    message=req_msg
                )
                
                req_timeout = request.timeout or 60 
                req_retries = max(request.retries or 0, 1)
                wait_timeout = req_timeout * req_retries
                executor = _Executor(request, logger=self._logger)
                while True:
                    try:
                        res = executor.wait(timeout=wait_timeout)
                    except:
                        raise RuntimeError(f'Timeout ({wait_timeout}) waiting for next response chunk')
                    yield res
                    if res.status == Status.Complete:
                        self._logger.info('Request successfully completed')
                        break
                    if res.status == Status.Error:
                        self._logger.info('Request completed with an error')
                        break
            except Exception as e:
                self._logger.error(f'Exception while processing request, err:{e}', exc_info=e)
                err_msg = '\n'.join(traceback.format_exception(e))
                err_msg += f'\n{e}'
                yield Response(
                    status=Status.Error,
                    message=f'Error while executing, err:\n{err_msg}'
                )

            finally: 
                executor.shutdown()
                self._lock.release()


def _request_to_string(req:Request) -> str:
    s = ''
    for key in dir(req):
        if key.startswith('_') or key[0].isupper() or key.endswith('_data'):
            continue
        v = getattr(req, key)
        s += f'{key}={v}\n'
    return s.rstrip()


class _Executor(threading.Thread):

    def __init__(
        self, 
        request:Request,
        logger:logging.Logger
    ):
        super().__init__(
            target=self._run, 
            name='DownloadRunExecutor', 
            daemon=True
        )
        self.logger = logger
        self.request = request
        self._shutdown_event = threading.Event()
        self._res_q = queue.Queue()
        self._line_buffer:str = ''
        self.start()

    def shutdown(self):
        self._shutdown_event.set()

    def wait(self, timeout:float) -> Response:
        return self._res_q.get(timeout=timeout)


    def _run(self):
        image_path:str = None
        extracted_path:str = None
        with tempfile.NamedTemporaryFile('wb', delete=False) as fd:
            image_path = fd.name
            self.logger.debug(f'Creating image file at {image_path}')
            fd.write(self.request.image_data)
           
        if self.request.image_path.endswith('.__image_archive__.zip'):
            extracted_path = f'{image_path}_extracted'
            self.logger.debug(f'Extracting {image_path} -> {extracted_path}')
            os.makedirs(extracted_path)
            extract_archive(image_path, extracted_path)
            image_path = extracted_path


        setup_script_path:str = None 
        if self.request.setup_script_data:
            with tempfile.NamedTemporaryFile('wb', delete=False) as fd:
                setup_script_path = fd.name
                self.logger.debug(f'Creating setup script file at {setup_script_path}')
                fd.write(self.request.setup_script_data)
            
        program_script_path:str = None 
        if self.request.program_script_data:
            with tempfile.NamedTemporaryFile('wb', delete=False) as fd:
                program_script_path = fd.name
                self.logger.debug(f'Creating program script file at {program_script_path}')
                fd.write(self.request.program_script_data)

        reset_script_path:str = None 
        if self.request.reset_script_data:
            with tempfile.NamedTemporaryFile('wb', delete=False) as fd:
                reset_script_path = fd.name
                self.logger.debug(f'Creating reset script file at {reset_script_path}')
                fd.write(self.request.reset_script_data)

        try:
            download_image_and_run(
                firmware_image_path=image_path,
                platform=self.request.platform or None,
                masserase=self.request.masserase,
                device=self.request.device or None,
                serial_number=self.request.serial_number or None,
                ip_address=self.request.ip_address or None,
                setup_script=setup_script_path,
                setup_script_args=self.request.setup_script_args or None,
                program_script=program_script_path,
                program_script_args=self.request.program_script_args or None,
                reset_script=reset_script_path,
                reset_script_args=self.request.reset_script_args or None,
                port=self.request.port or None,
                baud=self.request.baud or None,
                timeout=self.request.timeout or None,
                start_msg=self.request.start_msg or None,
                completed_msg=self.request.complete_msg or None,
                retries=self.request.retries,
                abort_event=self._shutdown_event,
                logger=self,
                outfile=self
            )
            self._post('Success', status=Status.Complete)

        except Exception as e:
            self.logger.exception('Exception in download_image_and_run', exc_info=e)
            err_msg = '\n'.join(traceback.format_exception(e))
            err_msg += f'\n{e}'
            self._post(err_msg, status=Status.Log)
            self._post(f'{e}', status=Status.Error)
        finally:
            try:
                os.remove(image_path)
            except:
                ...
            try:
                remove_directory(extracted_path)
            except:
                ...


    def _post(self, msg, status=Status.Running):
        res = Response(
            status=status,
            message=msg
        )
        self._res_q.put(res)


    def debug(self, msg:str):
        self.logger.debug(msg)
        self._post(
            msg,
            status=Status.DebugLog
        )

    def info(self, msg:str):
        self.logger.info(msg)
        self._post(
            msg,
            status=Status.Log
        )

    def error(self, msg:str):
        self.logger.error(msg)
        self._post(
            msg,
            status=Status.Log
        )

    def write(self, msg:str):
        self._line_buffer += msg

    def flush(self):
        if self._line_buffer:
            saved_terminators = None
            if hasattr(self.logger, 'set_terminator'):
                saved_terminators = self.logger.set_terminator('')
            self.logger.info(self._line_buffer)
            if saved_terminators:
                self.logger.set_terminator(saved_terminators)
            self._post(
                self._line_buffer,
                status=Status.SerialOut
            )
            self._line_buffer = ''
        

if __name__ == '__main__':
    _logger = get_logger('download_run', console=True, level='DEBUG')
    _server = DownloadRunServer(logger=_logger)

    try:
        _server.start()
    except KeyboardInterrupt:
        _server.stop()