from typing import Union 
import os 
import time

import http
from mltk.utils.network import find_listening_port
from mltk.utils.path import create_tempdir, fullpath
from mltk.utils.python import append_exception_msg



from .model import (
    MltkModel, 
    KerasModel,  
    load_mltk_model, 
    load_tflite_or_keras_model
)
from .tflite_model import TfliteModel
from .quantize_model import quantize_model
from .utils import (get_mltk_logger, ArchiveFileNotFoundError)



DEFAULT_PORT = 8080
DEFAULT_HOST = 'localhost'



def view_model(
    model: Union[str, MltkModel, KerasModel, TfliteModel], 
    host:str=None,
    port:int=None,
    test:bool=False,
    build:bool=False, 
    tflite:bool=False,
    timeout:float=7.0
):
    """View an interactive graph of the given model in a webbrowser

    .. seealso::
       * `Model Visualization Guide <https://siliconlabs.github.io/mltk/docs/guides/model_visualizer.html>`_
       * `Model visualization API examples <https://siliconlabs.github.io/mltk/mltk/examples/view_model.html>`_

    Args:
        model: Either 
        
            * a path to a `.tflite`, `.h5`, `.mltk.zip`, `.py` file,
            * or a :py:class:`mltk.core.MltkModel`, :py:class:`mltk.core.KerasModel`,
            * or :py:class:`mltk.core.TfliteModel` instance
        host: Optional, host name of local HTTP server
        port: Optional, listening port of local HTTP server
        test: Optional, if true load previously generated test model
        build: Optional, if true, build the MLTK model rather than loading previously trained model
        tflite: If true, view .tflite model otherwise view keras model
        timeout: Amount of time to wait before terminaing HTTP server
    """
    try:
        import netron 
    except:
        raise RuntimeError('Failed import netron Python package, try running: pip install netron OR pip install silabs-mltk[full]')

    # The default netron.server.ThreadedHTTPServer class that netron
    # uses inherits ThreadingMixIn which can hang.
    # Override this class to use http.server.ThreadingHTTPServer
    # which does not hang when it's shutdown
    netron.server.ThreadedHTTPServer = http.server.ThreadingHTTPServer
    netron.server._ThreadedHTTPServer = http.server.ThreadingHTTPServer

    logger = get_mltk_logger()

    model_path = _load_or_build_model(
        model, 
        tflite=tflite,
        build=build,
        test=test
    )
    logger.debug(f'Viewing model file: {model_path}')

    host = host or DEFAULT_HOST
    port = port or find_listening_port(default_port=DEFAULT_PORT)
    
    if os.getenv('MLTK_UNIT_TEST'):
        # Just return if we're doing a unit test
        return
    
    netron.start(file=model_path, address=(host, port), browse=True)
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            time.sleep(0.1)
        except KeyboardInterrupt:
            break 
    
    netron.stop()
    



def _load_or_build_model(
    model:Union[str, MltkModel, KerasModel, TfliteModel], 
    test:bool, 
    build:bool, 
    tflite:bool 
):
   
    if isinstance(model, KerasModel):
        model_path = f'{create_tempdir("tmp_models")}/model.h5'
        model.save(model_path, save_format='tf')
        return model_path

    if isinstance(model, TfliteModel):
        model_path = f'{create_tempdir("tmp_models")}/model.tflite'
        model.save(model_path)
        return model_path

    if isinstance(model, MltkModel):
        mltk_model = model 

    elif isinstance(model, str):
        if model.endswith(('.tflite', '.h5')):
            model_path = fullpath(model)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f'Model not found: {model_path}')
            return model_path

        else:
            mltk_model = load_mltk_model(
                model, 
                test=test, 
                print_not_found_err=True
            )
    else:
        raise ValueError('Invalid model argument')


    if build:
        if tflite:
            model_path = create_tempdir("tmp_models") + f'/{mltk_model.name}.tflite'
            quantize_model(
                model=mltk_model,
                build=True,
                output=model_path
            )
            return model_path
        else:
            keras_model = load_tflite_or_keras_model(mltk_model)
            model_path = f'{create_tempdir("tmp_models")}/model.h5'
            keras_model.save(model_path)
            return model_path

    else:
        try:
            if tflite:
                return mltk_model.tflite_archive_path
            else:
                return mltk_model.h5_archive_path
        except ArchiveFileNotFoundError as e:
            append_exception_msg(e, 
                '\nAlternatively, add the --build option to view the model without training it first'
            )
            raise

    
 
