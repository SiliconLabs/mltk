import typer

from mltk import cli


@cli.root_cli.command('compile')
def compile_model_command(
    model: str = typer.Argument(..., 
        help='''\b
One of the following:
- Name of MLTK model
- Path to trained model's archive (.mltk.zip)
- Path to MLTK model's python script
- Path to .tflite model
''',
        metavar='<model>'
    ),
    accelerator: str = typer.Option(..., '--accelerator', '-a', 
        help='Name of accelerator',
         metavar='<name>'
    ),
    verbose: bool = typer.Option(False, '--verbose', '-v', 
        help='Enable verbose console logs'
    ),
    output: str = typer.Option(None, '--output', '-o', 
        help='''\b
One of the following:
- Path to generated output .tflite file
- Directory where output .tflite is generated
- If omitted, .tflite is generated in the same directory as the given model and the model archive is updated (if an mltk model is provided)''',
        metavar='<path>'
    ),
):
    """Compile a model for the specified accelerator

    """

    # Import all required packages here instead of at top
    # to help improve the CLI's responsiveness
    from mltk.core import (
        compile_model,
        load_mltk_model
    )


    logger = cli.get_logger(verbose=verbose)

    if not model.endswith('.tflite'):
        try:
            model = load_mltk_model(
                model,  
                print_not_found_err=True
            )
        except Exception as e:
            cli.handle_exception('Failed to load model', e)

    try:
        tflite_path = compile_model(
            model, 
            accelerator=accelerator,
            output=output
        )
    except Exception as e:
        cli.handle_exception('Failed to compile model', e)

    if output:
        logger.info(f'Generated model at {tflite_path}')

