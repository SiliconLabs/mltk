import logging

from mltk import MLTK_ROOT_DIR
from mltk.utils.cmake import build_mltk_target


def build_tflite_micro_wrapper(
    clean:bool=True, 
    verbose:bool=False,
    logger:logging.Logger=None,
    use_user_options=False,
    debug:bool=False,
):  
    """Build the TF-Lite Micro Python wrapper for the current OS/Python environment"""
    logger = logger or logging.getLogger()

    build_mltk_target(
        target='mltk_tflite_micro_wrapper',
        build_subdir='tflm_wrap',
        source_dir=MLTK_ROOT_DIR,
        clean=clean,
        verbose=verbose,
        debug=debug,
        logger=logger,
        use_user_options=use_user_options,
    )
