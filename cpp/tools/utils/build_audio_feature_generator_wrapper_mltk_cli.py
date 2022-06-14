
import logging
import typer


from cpp.audio_feature_generator_wrapper import build_audio_feature_generator_wrapper
from mltk import cli 


@cli.build_cli.command('audio_feature_generator_wrapper')
def build_tflite_micro_wrapper_command(
    verbose: bool = typer.Option(False, '--verbose', '-v', 
        help='Enable verbose console logs'
    ),
    clean: bool = typer.Option(True, 
        help='Clean the build directory before building'
    ),
    use_user_options: bool = typer.Option(False, '--user', '-u',
        help='Use the <mltk>/user_options.cmake file while building the wrapper. If omitted then this file is IGNORED'
    ),
    debug: bool = typer.Option(False, '--debug', '-d',
        help='Build debug version of tflite wrapper')
):
    """Build the AudioFeatureGenerator Python wrapper
    
    \b
    This builds the AudioFeatureGenerator Python wrapper:  
    https://github.com/siliconlabs/mltk/tree/master/cpp/audio_feature_generator_wrapper
    \b
    NOTE: The built wrapper library is copied to:
    https://github.com/siliconlabs/mltk/tree/master/mltk/core/preprocess/audio/audio_feature_generator

    """

    logger = cli.get_logger(verbose=verbose)

    try:
        build_audio_feature_generator_wrapper(
            logger=logger,
            clean=clean,
            verbose=verbose,
            use_user_options=use_user_options,
            debug=debug
        )
    except Exception as e:
        cli.handle_exception('Failed to build audio_feature_generator_wrapper', e)

    logger.info('Done')

