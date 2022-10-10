
import os
import io
import logging
from typing import Union



from mltk.utils.string_formatting import format_units
from mltk.utils.path import fullpath
from mltk.utils.python import append_exception_msg

from .model import (
    MltkModel, 
    KerasModel,  
    load_mltk_model, 
    load_tflite_or_keras_model
)

from .utils import ArchiveFileNotFoundError, get_mltk_logger
from .model.metrics import calculate_model_metrics
from .tflite_model import TfliteModel

def summarize_model(
    model: Union[str, MltkModel, KerasModel, TfliteModel], 
    tflite:bool=False,
    build:bool=False,
    test:bool=False,
    built_model:Union[KerasModel, TfliteModel]=None
) -> str:
    """Generate a summary of the given model 
    and return the summary as a string

    .. seealso::
       * `Model Summary Guide <https://siliconlabs.github.io/mltk/docs/guides/model_summary.html>`_
       * `Model summary API examples <https://siliconlabs.github.io/mltk/mltk/examples/summarize_model.html>`_


    Args:
        model: Either 
        
            * a path to a `.tflite`, `.h5`, `.mltk.zip`, `.py` file,
            * or a :py:class:`mltk.core.MltkModel`, :py:class:`mltk.core.KerasModel`,
            * or :py:class:`mltk.core.TfliteModel` instance
        tflite: If true, the return the summary of the corresponding `.tflite model`.
            If true and model= :py:class:`mltk.core.KerasModel`, this will quantize it into a `.tflite` model
        build: If true, then generate a `.h5` or `.tflite` by training the given :py:class:`mltk.core.MltkModel` model for 1 epoch.
            This is useful for summarizing the :py:class:`mltk.core.MltkModel` without fully training the model first
        test: If true and the model is the name of a MltkModel, then load the MltkModel in testing mode
        built_model: Optional, previously built :py:class:`mltk.core.KerasModel` or
            :py:class:`mltk.core.TfliteModel` associated with given :py:class:`mltk.core.MltkModel`

    Returns:
        A summary of the given model as a string

    """
    mltk_model = None
    mltk_model_summary = None 
    tflite_size = None

    logger = get_mltk_logger()

    try:
        mltk_model, built_model = _load_or_build_model(
            model,
            tflite=tflite,
            build=build,
            built_model=built_model,
            test=test,
            logger=logger
        )
    except ArchiveFileNotFoundError as e:
        append_exception_msg(e, 
            '\nAlternatively, add the --build option to summarize the model without training it first'
        )
        raise


    if isinstance(built_model, TfliteModel):
        tflite_size = built_model.flatbuffer_size
        if mltk_model is None:
            # If no MLTK model was given (because we directly loaded a .tflite file)
            # then create a default model and attempt to load the common metadata entries
            mltk_model = MltkModel()

        mltk_model.deserialize_tflite_metadata(built_model)
    

    if mltk_model is not None:
        mltk_model_summary = mltk_model.summary()

    model_metrics = calculate_model_metrics(built_model, logger=logger)

    summary = ''
    if isinstance(built_model, KerasModel):
        string_buffer = io.StringIO()
        def _writeln(s):
            string_buffer.write(s + '\n')
        built_model.summary(print_fn=_writeln)
        summary += string_buffer.getvalue()
    else:
        summary += built_model.summary()

    summary += '\n'
    summary += f'Total MACs: {format_units(model_metrics["total_macs"])}\n'
    summary += f'Total OPs: {format_units(model_metrics["total_ops"])}\n'
    
    if mltk_model_summary is not None:
        summary += f'{mltk_model_summary}\n'

    if tflite_size:
        summary += f'.tflite file size: {format_units(tflite_size, precision=1, add_space=False)}B\n'

    return summary.strip()




def _load_or_build_model(
    model:Union[str, MltkModel, KerasModel, TfliteModel], 
    built_model:Union[KerasModel, TfliteModel],
    test:bool,
    tflite:bool,
    build:bool,
    logger:logging.Logger
):
    """Load a previously trained .tflite/.h5 OR build the model now"""
    mltk_model = None 


    # If a MltkModel instance was given
    if isinstance(model, MltkModel):
        mltk_model = model 

    # Elif if a KerasModel instance was given
    elif isinstance(model, KerasModel):
        built_model = model 

     # Elif if a KerasModel instance was given
    elif isinstance(model, TfliteModel):
        built_model = model   

    elif not isinstance(model, str):
        raise Exception('model argument must be a string or MltkModel,KerasModel,TfliteModel instance')  

    # Else if the path to a .h5 or .tflite was given
    elif model.endswith(('.tflite', '.h5')):
        model_path = fullpath(model)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model not found: {model_path}')

        built_model = load_tflite_or_keras_model(model_path)

    # Else a MLTK model name was given
    else:
        mltk_model = load_mltk_model(
            model,  
            test=test,
            print_not_found_err=True
        )

    if build and mltk_model is None:
        raise Exception('Must provide MltkModel with the build option')


    # If we have a MltkModel instance but no built model instance
    if mltk_model is not None and built_model is None:
        # If we want to build keras or .tflite model from the .tflite
        # (i.e. if the model has not already been trained)
        if build:
            from .train_model import train_model
            from .quantize_model import quantize_model

            if tflite:
                built_model = quantize_model(
                    model=mltk_model,
                    build=True,
                    output='tflite_model'
                )
            else:
                built_mltk_model = load_mltk_model(mltk_model.model_specification_path, test=True)
                results = train_model(
                    model=built_mltk_model,
                    epochs=1,
                    quantize=False,
                    clean=None,
                    create_archive=False,
                    verbose=logger.verbose,
                )
                built_model = results.keras_model

        # Else if we need to load the .tflite from the MltkModel archive
        elif tflite:
            # If a .tflite path was given
            if isinstance(tflite, str):
                built_model = load_tflite_or_keras_model(
                    tflite, 
                )
            # Else load the .tflite from the model archive
            else:
                built_model = load_tflite_or_keras_model(
                    mltk_model, 
                    model_type='tflite',
                )

        # Else load the .h5 from the MltkModel archive
        else:
            built_model = load_tflite_or_keras_model(
                mltk_model, 
                model_type='h5',
            )

    return mltk_model, built_model