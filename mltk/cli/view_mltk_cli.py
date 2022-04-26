
import typer

from mltk import cli



@cli.root_cli.command('view')
def view_model_command(
    model: str = typer.Argument(..., 
        help='''\b
One of the following:
- Path to .tflite model file
- Path to .h5 model file
- Name of MLTK model
- Path to model's archive (.mltk.zip)
- Path to MLTK model's python script''',
        metavar='<model>'
    ),
    tflite: bool = typer.Option(False, '--tflite', 
        help='View the .tflite model file in the MLTK model\'s archive, or if the --build option is given, generate .tflite file before viewing'
    ),
    build: bool = typer.Option(False, '--build', '-b', 
        help='Build the  model rather than loading from a pre-trained file in the MLTK model archive'
    ),
    host:str = typer.Option(None, '-h', '--host',
        help='Local interface to start HTTP server',
        metavar='<host>'
    ),
    port:int = typer.Option(8080, '-p', '--port',
        help='Listen port of HTTP server used to view graph',
        metavar='<port>'
    ),
    verbose: bool = typer.Option(False, '--verbose', '-v', 
        help='Enable verbose console logs'
    )
):
    """View an interactive graph of the given model in a webbrowser

    This is based on the utility: https://netron.app

    \b
    ----------
     Examples
    ----------
    \b
    # View pre-trained Keras model
    mltk view image_example1
    \b
    # View pre-trained tflite model
    mltk view image_example1 --tflite
    \b
    # View provided .tflite model file
    mltk view ~/workspace/my_model.tflite
    \b
    # Generate the .tflite then view it
    # MLTK model image_example1 need not be trained first
    mltk view image_example1 --tflite --build

    """
    # Import all required packages here instead of at top
    # to help improve the CLI's responsiveness
    from mltk.core import view_model


    logger = cli.get_logger(verbose=verbose)

    if not verbose:
        logger.console_level = 'WARNING'

    try:
        view_model(
            model=model, 
            tflite=tflite,
            build=build, 
            host=host, 
            port=port
        )
    except Exception as e:
        cli.handle_exception('Failed to view model', e)

 