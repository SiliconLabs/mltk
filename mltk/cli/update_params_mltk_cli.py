import os
import json

import yaml
import typer

from mltk import cli


@cli.root_cli.command('update_params', cls=cli.AdditionalArgumentParsingCommand)
def update_params_command(
    ctx: typer.Context,
    model: str = typer.Argument(..., 
        help='''\b
One of the following:
- Name of trained MLTK model
- Path to trained model's archive (.mltk.zip)
- Path to .tflite model file''',
        metavar='<model>'
    ),
    params_path: str = typer.Option(None, '--params', '-p',
        help='Optional path to .json or .yaml file contains parameters to add to given model',
        metavar='<params path>'
    ),
    description: str = typer.Option(None, '--description', '-d',
        help='Optional description to add to the generated .tflite',
        metavar='<description>'
    ),
    verbose: bool = typer.Option(False, '--verbose', '-v', 
        help='Enable verbose console logs'
    ),
    output: str = typer.Option(None, '--output', '-o', 
        help='''\b
One of the following:
- Path to generated output .tflite file
- Directory where output .tflite is generated
- If omitted, .tflite is generated in the MLTK model's log directory and the model archive is updated (if applicable)''',
        metavar='<path>'
    ),
    accelerator: str = typer.Option(None, '--accelerator', '-a', 
        help='Optional accelerator to use when calculating the "runtime_memory_size" model parameter. If omitted then use the CMSIS kernels',
        metavar='<accelerator>'
    ),
    update_device:bool = typer.Option(False, '-d', '--device',
        help='''\b
If provided, program the updated .tflite to end of the flash memory of the the connected device.
Supported apps (e.g. model_profiler, audio_classifier, etc) will use this .tflite instead of the default model.
This allows for making changes to the model without re-building the firmware application.
If this option is provided, then the device must be locally connected'''
    ),
):
    """Update the parameters of a previously trained model

    \b
    This updates the metadata of a previously trained .tflite model.
    The parameters are taken from either the given MltkModel's python script 
    or the given "params" .json/.yaml file. Additional int/float/str parameters can also be
    given on the command line (see examples below).
    \b
    NOTE: The .tflite metadata is only modified. 
    The weights and model structure of the .tflite file are NOT modified. 
    
    \b
    ----------
     Examples
    ----------
    \b
    # Update the tflite_micro_speech model parameters with any modifications
    # made in mltk/models/tflite_micro/tflite_micro_speech.py
    # The associated model archive is updated with the updated .tflite
    mltk update_params tflite_micro_speech
    \b
    # Update my_model.tflite with parameters in my_params.json
    # Also update the model description
    mltk update_params my_model.tflite --params my_params.json --description "My model is great!"
    \b
    # Update my_model.tflite with additional params on the command-line
    mltk update_params my_model.tflite my_custom_param="some value" led_period_ms=43

    """
    # Import all required packages here instead of at top
    # to help improve the CLI's responsiveness
    from mltk.core import (
        TfliteModel, 
        TfliteModelParameters, 
        load_mltk_model, 
        update_model_parameters
    )
    from mltk.utils.path import fullpath 
    from mltk.utils.firmware_apps import program_model


    logger = cli.get_logger(verbose=verbose)
    model_arg = model

    if not model.endswith(('.tflite', '.mltk.zip')):
        try:
            model = load_mltk_model(
                model,  
                print_not_found_err=True
            )
        except Exception as e:
            cli.handle_exception('Failed to load model', e)


    ###################################################################
    def _load_params_file(params_path:str, params:dict):
        """Load model parameters from file"""

        if params_path is None:
            return

        params_path = fullpath(params_path)
        if not os.path.exists(params_path):
            raise FileNotFoundError(f'Not found: {params_path}')

        if params_path.endswith('.json'):
            with open(params_path, 'r') as fp:
                d = json.load(fp)
                if not isinstance(d, dict):
                    raise Exception('.json file must contain an object at the root')
                params.update(d)

        elif params_path.endswith(('.yaml', '.yml')):
            with open(params_path, 'r') as fp:
                d = yaml.load(fp, Loader=yaml.SafeLoader)
                if not isinstance(d, dict):
                    raise Exception('.yaml file must contain an object at the root')
                params.update(d)

        else:
            raise Exception('--params option must be a path to a .json or .yaml file')


    ###################################################################
    def _load_cli_params(additional_params:dict, params:dict):
        """Load model parameters from command-line options"""

        for key, value in additional_params.items():
            # Try to convert the value
            # to an integer or float
            # other default to a string
            try:
                value = int(value)
            except:
                try:
                    value = float(value)
                except:
                    pass 

            params[key] = value



    ###################################################################
    #
    # Actual command logic
    #
    params = {}
    try:
        _load_params_file(params_path, params=params)
    except Exception as e:
        cli.handle_exception('Failed to load model', e)

    try:
        _load_cli_params(ctx.meta['additional_variables'], params=params)
    except Exception as e:
        cli.handle_exception('Failed to parse cli parameters', e)


    try:
        tflite_path = update_model_parameters(
            model, 
            params=params,
            description=description,
            output=output,
            accelerator=accelerator
        )
    except Exception as e:
        cli.handle_exception('Failed to update model parameters', e)

    try:
        tflite_model = TfliteModel.load_flatbuffer_file(tflite_path)
        model_params = TfliteModelParameters.load_from_tflite_file(tflite_path)
        logger.info(f'Description: {tflite_model.description}\n{model_params}')
    except Exception as e:
        logger.debug('Failed to summarize model', exc_info=e)
        cli.print_warning(f'Failed to summarize model, err: {e}')

    logger.info(f'Updated model parameters: {model_arg if model_arg.endswith(".mltk.zip") else tflite_path}')


    if update_device:
        program_model(
            tflite_model=tflite_model,
            logger=logger,
            halt=False,
        )