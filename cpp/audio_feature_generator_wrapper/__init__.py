import logging


from mltk.utils.cmake import build_mltk_target


def build_audio_feature_generator_wrapper(
    clean:bool=True, 
    verbose:bool=False,
    logger:logging.Logger=None
):
    """Build the GSDK AudioFeatureGenerator Python wrapper for the current OS/Python environment"""
    build_mltk_target(
        target='mltk_audio_feature_generator_wrapper',
        build_subdir='afg_wrap',
        clean=clean,
        verbose=verbose,
        logger=logger
    )