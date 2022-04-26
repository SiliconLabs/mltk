import logging


from mltk.utils.cmake import build_mltk_target


def build_mvp_wrapper(
    clean:bool=True, 
    verbose:bool=False,
    logger:logging.Logger=None
):
    """Build the MVP Python wrapper for the current OS/Python environment"""
    build_mltk_target(
        target='mltk_mvp_wrapper',
        build_subdir='mvp_wrap',
        clean=clean,
        verbose=verbose,
        logger=logger
    )