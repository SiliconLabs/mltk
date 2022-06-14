
import os
from typing import Union


from .tflite_model import TfliteModel
from .tflite_micro import TfliteMicro
from .model import (
    MltkModel, 
    load_mltk_model,
)
from .utils import get_mltk_logger



def compile_model(
    model:Union[MltkModel, TfliteModel, str],
    accelerator:str,
    output:str=None,
    update_archive:bool=None,
) -> Union[str,TfliteModel]:
    """Compile the given quantized .tflite model for the specified accelerator
    
    Returns:
        The file path to the compiled `.tflite` OR TfliteModel object if output='tflite_model'
    """

    mltk_model = None

    if isinstance(model, TfliteModel):
        tflite_model = model

    elif isinstance(model, MltkModel):
        mltk_model = model
        tflite_model = TfliteModel.load_flatbuffer_file(model.tflite_archive_path)

    elif isinstance(model, str):
        if model.endswith('.tflite'):
            tflite_model = TfliteModel.load_flatbuffer_file(model)
        elif model.endswith('.h5'):
            raise ValueError(
                'Must provide path to quantized .tflite model file'
            )
        else:
            mltk_model = load_mltk_model(model)
            tflite_model = TfliteModel.load_flatbuffer_file(mltk_model.tflite_archive_path)

    else:
        raise ValueError(
            'Must provide path to .tflite, TfliteModel instance, MltkModel instance, name of MLTK model, or path to '
            'model archive (.mltk.zip) or specification script (.py)'
        )

   
    if not TfliteMicro.accelerator_is_supported(accelerator):
        raise ValueError(f'Unknown accelerator: {accelerator}, supported accelerators are: {", ".join(TfliteMicro.get_supported_accelerators())}')

    accelerator = TfliteMicro.normalize_accelerator_name(accelerator).lower()

    logger = get_mltk_logger()

    tflm_accelerator = TfliteMicro.get_accelerator(accelerator)
    if not tflm_accelerator.supports_model_compilation:
        raise RuntimeError(f'Accelerator {accelerator} does not support compilation')

    compiled_tflite_model = tflm_accelerator.compile_model(
        tflite_model,
        logger=logger 
    )

    # Determine if we should update the model archive with the generated .tflite
    if update_archive is None:
        update_archive = False
        if not output and mltk_model is not None:
            update_archive = mltk_model.check_archive_file_is_writable()
    if update_archive and mltk_model is None:
        raise ValueError('Must provide MltkModel if updating archive')


    tflite_path = tflite_model.path or 'my_model.tflite'
    model_name = os.path.basename(tflite_path)[:-len('.tflite')]

    # Determine the return value of this API
    if output:
        if output == 'tflite_model':
            retval = 'tflite_model'
        elif output.endswith('.tflite'):
            retval = output
        else:
            retval = f'{output}/{model_name}.{accelerator}.tflite'
    elif mltk_model is not None:
        retval = f'{mltk_model.log_dir}/{mltk_model.name}.{accelerator}.tflite'
    else:
        retval = f'{tflite_path[:-len(".tflite")]}.{accelerator}.tflite'


    if retval == 'tflite_model':
        retval = compiled_tflite_model
    else:
        logger.info(f'Saving {retval}')
        compiled_tflite_model.save(retval)
        if update_archive:
            logger.info(f'Updating {mltk_model.archive_path}')
            mltk_model.add_archive_file(retval)


    return retval