
import logging
import typer


from cpp.mvp_wrapper import build_mvp_wrapper
from mltk import cli 



@cli.build_cli.command('mvp_wrapper')
def build_mvp_wrapper_command(
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
    """Build the MVP-accelerated Tensorflow-Lite Micro Kernels & MVP simulator Python wrapper

    \b
    This builds the MVP-accelerated Tensorflow-Lite Micro kernels and MVP simulator Python wrapper:  
    https://github.com/siliconlabs/mltk/tree/master/cpp/mvp_wrapper
    \b
    NOTE: The built wrapper library is copied to:
    https://github.com/siliconlabs/mltk/tree/master/mltk/core/tflite_micro/accelerators/mvp

    """

    logger = cli.get_logger(verbose=verbose)

    try:
        build_mvp_wrapper(
            logger=logger,
            clean=clean,
            verbose=verbose,
            use_user_options=use_user_options,
            debug=debug
        )
    except Exception as e:
        cli.handle_exception('Failed to build mvp_wrapper', e)

    logger.info('Done')


