import os
import typer

from mltk import cli



@cli.root_cli.command('summarize')
def summarize_model_command(
    model: str = typer.Argument(..., 
        help='''\b
One of the following:
- Path to .tflite model file
- Path to .h5 model file
- Name of MLTK model
- Path to trained model's archive (.mltk.zip)
- Path to MLTK model's python script''',
        metavar='<model>'
    ),
    tflite: bool = typer.Option(False, '--tflite', 
        help='''Summarize the .tflite model file in the MLTK model's archive, or if the --build option is given, generate .tflite file before summarizing'''
    ),
    build: bool = typer.Option(False, '--build', '-b', 
        help='Build the model rather than loading from a pre-trained file in the MLTK model archive'
    ),
    output: str = typer.Option(None, '--output', '-o', 
        help='File path of generated summary file. If omitted, the summary is printed to console',
        metavar='<path>'
    ),
    verbose: bool = typer.Option(False, '--verbose', '-v', 
        help='Enable verbose console logs'
    )
):
    """Generate a summary of a model
    
    \b
    If a .h5 file is provided or a MLTK model name/archive/script and *no* --tflite option,
    then print a summary of the Keras Model.
    \b
    If a .tflite file is provided or a MLTK model name/archive/script and the --tflite option,
    then print a summary of the .tflite model.
    \b
    Use the --build option if the model has not been previously trained.

    \b
    ----------
     Examples
    ----------
    \b
    # Print a summary of pre-trained Keras model
    mltk summarize audio_example1
    \b
    # Print a summary of pre-trained TF-Lite model
    mltk summarize audio_example1 --tflite
    \b
    # Generate a .tflite then print a summary
    # In this case, the model need not be previously trained
    mltk summarize audio_example1 --build --tflite
    \b
    # Print of summary of the given .tflite
    mltk summarize some/path/my_model.tflite
    \b
    # Print of summary of the given model archive's .tflite
    mltk summarize some/path/my_model.mltk.zip --tflite
    """

    # Import all required packages here instead of at top
    # to help improve the CLI's responsiveness
    from mltk.core import summarize_model


    logger = cli.get_logger(verbose=verbose)

    if not verbose:
        saved_console_level = logger.console_level
        logger.console_level = 'WARNING'
    
    summary = summarize_model(
        model=model,
        tflite=tflite,
        build=build
    )

    if not verbose:
        logger.console_level = saved_console_level

    if output:
        out_dir = os.path.dirname(output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output, 'w') as f:
            f.write(summary)
        
    else:
        logger.info(f'{summary}')


