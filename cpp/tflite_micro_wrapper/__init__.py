import logging
from mltk.utils.cmake import build_mltk_target


def build_tflite_micro_wrapper(
    clean:bool=True, 
    verbose:bool=False,
    logger:logging.Logger=None
):
    """Build the TF-Lite Micro Python wrapper for the current OS/Python environment"""
    build_mltk_target(
        target='mltk_tflite_micro_wrapper',
        build_subdir='tflm_wrap',
        clean=clean,
        verbose=verbose,
        logger=logger
    )