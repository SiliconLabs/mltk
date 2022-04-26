
import logging
import typer

from mltk import cli, MLTK_ROOT_DIR
from mltk.utils.cmake import build_mltk_target


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


def build_mvp_wrapper(
    clean:bool=False, 
    verbose:bool=True,
    logger:logging.Logger=None,
    use_user_options=False,
    debug:bool=False,
):  
    """Build the MVP kernels + simulator Python wrapper for the current OS/Python environment"""
    logger = logger or logging.getLogger()


    build_mltk_target(
        target='mltk_mvp_wrapper',
        build_subdir='mvp_wrap',
        source_dir=MLTK_ROOT_DIR,
        clean=clean,
        verbose=verbose,
        debug=debug,
        logger=logger,
        use_user_options=use_user_options,
    )
