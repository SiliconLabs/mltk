import logging 
import threading
import sys 
import os
import tempfile
import shutil
from typing import Iterable, TextIO

from concurrent.futures import ThreadPoolExecutor


try:
    import grpc 
except ModuleNotFoundError:
    print('Must install "grpc" python package')
    print('pip install grpc')
    sys.exit(-1)


from mltk import cli
from mltk.utils.logger import get_logger, make_filelike



_curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_curdir, 'proto'))
if not __package__:                  
    sys.path.insert(0, os.path.dirname(_curdir))
    __package__ = os.path.basename(_curdir) # pylint: disable=redefined-builtin


from .proto .download_run_pb2 import (Request, Response, Status) # pylint: disable=no-name-in-module
from .proto.download_run_pb2_grpc import DownloadRunStub



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
    remote_address:str='localhost:50051',
    lock_timeout:float = 60.0,
    outfile:TextIO = None
):
    logger = logger or get_logger()
    if not outfile:
        make_filelike(logger)
        outfile = logger
    logger.debug(f'Connecting to remote server: {remote_address}')

    if os.path.isfile(firmware_image_path):
        logger.debug(f'Reading firmware image {firmware_image_path}')
        with open(firmware_image_path, 'rb') as f:
            image_data = f.read()
    elif os.path.isdir(firmware_image_path):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            firmware_archive = f.name + '.__image_archive__.zip'

        logger.debug(f'Archiving firmware directory {firmware_image_path} -> {firmware_archive}')
        shutil.make_archive(firmware_archive[:-len('.zip')], 'zip', firmware_image_path, base_dir=firmware_image_path)
    
        with open(firmware_archive, 'rb') as f:
            image_data = f.read()
            firmware_image_path = firmware_archive
    else:
        raise ValueError(f'Invalid firmware path: {firmware_image_path}')

    setup_script_data = None
    if setup_script:
        logger.debug(f'Reading setup script {setup_script}')
        with open(setup_script, 'rb') as f:
            setup_script_data = f.read() 

    program_script_data = None
    if program_script:
        logger.debug(f'Reading program script {program_script}')
        with open(program_script, 'rb') as f:
            program_script_data = f.read() 

    reset_script_data = None
    if reset_script:
        logger.debug(f'Reading reset script {reset_script}')
        with open(reset_script, 'rb') as f:
            reset_script_data = f.read() 


    request = Request(
         image_data=image_data,
         image_path=firmware_image_path,
         platform=platform,
         masserase=masserase,
         device=device,
         serial_number=serial_number,
         ip_address=ip_address,
         setup_script_data=setup_script_data,
         setup_script_args=setup_script_args,
         program_script_data=program_script_data,
         program_script_args=program_script_args,
         reset_script_data=reset_script_data,
         reset_script_args=reset_script_args,
         port=port,
         baud=baud,
         timeout=timeout,
         start_msg=start_msg,
         complete_msg=completed_msg,
         retries=retries,
         lock_timeout=lock_timeout,
    )

    shutdown_event = threading.Event()

    logger.debug('Sending request to remote server ....')
    executor =  _Executor(
        address=remote_address,
        logger=logger,
        outfile=outfile,
        request=request,
        shutdown_event=shutdown_event
    )

    try:
        err_msg = executor.wait()
        if err_msg:
            raise RuntimeError(err_msg)
    except KeyboardInterrupt:
        pass 
    finally:
        shutdown_event.set()


class _Executor:

    def __init__(
        self,
        address:str,
        request:Request,
        logger:logging.Logger,
        outfile:TextIO,
        shutdown_event:threading.Event
    ):
        self.logger = logger 
        self.outfile = outfile
        self.address = address
        self.request = request
        self.shutdown_event = shutdown_event
        self._stub:DownloadRunStub = None


    def wait(self) -> str:
        executor = ThreadPoolExecutor()
        with grpc.insecure_channel(
            self.address,
            compression=grpc.Compression.Gzip,
            options=[('grpc.enable_http_proxy', 0)]
        ) as channel:
            self._stub = DownloadRunStub(channel)
            future = executor.submit(
                self._process
            )
            return future.result()


    def _process(self) -> str:
        try:
            response_iterator:Iterable[Response] = self._stub.DownloadAndRun(self.request)
            for resp in response_iterator:
                if resp.status == Status.DebugLog:
                    self.logger.debug(resp.message)
                elif resp.status == Status.Log:
                    self.logger.info(resp.message)

                elif resp.status == Status.SerialOut:
                    self._write_serial_out(resp.message)
    
                elif resp.status == Status.Complete:
                    return 
                
                elif resp.status == Status.Timeout:
                    return 'Timed-out waiting for the request to start on the remote server'
                
                elif resp.status == Status.Error:
                    return f'The request failed on the remote server: {resp.message}'

                if self.shutdown_event.is_set():
                    return

        except Exception as e:
            self.logger.exception(f'\n{e}', exc_info=e)
            return f'{e}'
            
        finally:
            self.shutdown_event.set()

        self.logger.debug('\nRequest successfully completed')


    def _write_serial_out(self, msg:str):
        saved_terminators = None
        if hasattr(self.outfile, 'set_terminator'):
            saved_terminators = self.outfile.set_terminator('')
        self.outfile.write(msg)
        self.outfile.flush()
        if saved_terminators:
            self.outfile.set_terminator(saved_terminators)


if __name__ == '__main__':
    download_image_and_run(
        f'{_curdir}/../../../../../mltk_internal/build/arm-gcc/Release/mltk_model_profiler.s37',
        platform='fpga_npu',
        logger=cli.get_logger(verbose=True),
        remote_address='lab0015144:50051',
        serial_number='440204659',
        port='COM4'
    )