import os
from setuptools.command.build_ext import build_ext



from cpp.mvp_wrapper import build_mvp_wrapper
from cpp.audio_feature_generator_wrapper import build_audio_feature_generator_wrapper
from cpp.tflite_micro_wrapper import build_tflite_micro_wrapper



from .utils import get_command_logger


class BuildWrappersCommand(build_ext):
    """Custom setup.py command to build the Python C++ wrappers
    
    Invoke this command by:

    ```python
    python setup.py build_ext [--afg] [--tflite-micro] [--mvp]
    ```

    """
    editable_mode = True
    description = 'Build C++ Python wrappers'
    user_options = [
        # The format is (long option, short option, description).
        ('afg', None, 'Build the AudioFeatureGenerator Python wrapper'),
        ('tflite-micro', None, 'Build the TF-Lite Micro wrapper Python wrapper'),
        ('mvp', None, 'Build the MVP hardware simulator Python wrapper'),
        ('no-clean', None, 'Do NOT clean any previous build artifacts before building'),
    ]

    def initialize_options(self):
        """Set default values for options."""
        self.afg = False 
        self.tflite_micro = False
        self.mvp = False 
        self.verbose = os.getenv('MLTK_VERBOSE_INSTALL', '0') == '1'
        self.no_clean = False

    def finalize_options(self):
        """Post-process options."""
        # If no options were specified, then build everything
        if not (self.afg or self.tflite_micro or self.mvp):
            self.afg = True 
            self.tflite_micro = True
            self.mvp = True
        # else if we're building the mvp wrapper, then build the tflite_micro wrapper first
        elif self.mvp:
            self.tflite_micro = True


    def run(self):
        """Run command."""
        logger = get_command_logger(self)
 
        if self.tflite_micro:
            logger.info('#' * 80)
            logger.info('Building TF-Lite Micro Python Wrapper ...')
            build_tflite_micro_wrapper(
                verbose=self.verbose,
                clean=not self.no_clean,
                logger=logger
            )
        if self.mvp:
            logger.info('#' * 80)
            logger.info('Building MVP Python Wrapper ...')
            build_mvp_wrapper(
                verbose=self.verbose,
                clean=not self.no_clean,
                logger=logger
            )
        if self.afg:
            logger.info('#' * 80)
            logger.info('Building AudioFeatureGenerator Python Wrapper ...')
            build_audio_feature_generator_wrapper(
                verbose=self.verbose,
                clean=not self.no_clean,
                logger=logger
            )