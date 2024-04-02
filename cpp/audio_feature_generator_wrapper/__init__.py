import logging

from mltk import MLTK_ROOT_DIR


def build_audio_feature_generator_wrapper(
    clean:bool=True, 
    verbose:bool=False,
    logger:logging.Logger=None,
    use_user_options=False,
    debug:bool=False,
):  
    """Build the AudioFeatureGenerator Python wrapper for the current OS/Python environment"""
    # This must be import here in case the CMake package is installed during setup
    from mltk.utils.cmake import build_mltk_target
    
    logger = logger or logging.getLogger()

    build_mltk_target(
        target='mltk_audio_feature_generator_wrapper',
        build_subdir='afg_wrap',
        source_dir=MLTK_ROOT_DIR,
        clean=clean,
        verbose=verbose,
        debug=debug,
        logger=logger,
        use_user_options=use_user_options,
    )
