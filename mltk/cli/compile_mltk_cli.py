import typer

from mltk import cli


@cli.root_cli.command('compile', context_settings=dict(allow_extra_args=True, ignore_unknown_options=True))
def compile_model_command(
    ctx: typer.Context,
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
        load_tflite_model
    )


    logger = cli.get_logger(verbose=verbose)

    if not model.endswith('.tflite'):
        try:
            model = load_tflite_model(
                model,  
                print_not_found_err=True
            )
        except Exception as e:
            cli.handle_exception('Failed to load model', e)

    kwargs = get_additional_options(ctx)

    try:
        tflite_path = compile_model(
            model, 
            accelerator=accelerator,
            output=output,
            logger=logger,
            **kwargs
        )
    except Exception as e:
        cli.handle_exception('Failed to compile model', e)

    if output:
        logger.info(f'Generated model at {tflite_path}')



def get_additional_options(ctx: typer.Context) -> dict:
    retval = dict()
    args = ctx.args
    while args:
        if args[0].startswith('--'):
            if len(args) < 2:
                raise ValueError(f'{args[0]} option must have value after it')
            key = args[0][2:].replace('-', '_')
            value = args[1]
            try:
                value = int(value)
            except:
                pass
            retval[key] = value
            args = args[2:]

        else:
            args = args[1:]

    return retval
