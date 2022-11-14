
import logging
from mltk.core.profiling_results import ProfilingModelResults
from mltk.core.tflite_model import TfliteModel


class TfliteMicroAccelerator:
    """TF-Lite Micro Accelerator
    
    This class allows for providing hardware-accelerated
    kernels to the TFLM interpreter.
    """

    def __init__(self, accelerator_wrapper):
        self._accelerator_wrapper = accelerator_wrapper


    @property
    def name(self) -> str:
        """The name of the accelerator"""
        return self._accelerator_wrapper.name() 

    @property
    def api_version(self) -> int:
        """The API version number this wrapper was built with
        This number must match the tflite_micro_wrapper's API version
        """
        return self._accelerator_wrapper.api_version()

    @property
    def git_hash(self) -> int:
        """Return the GIT hash of the MLTK repo used to compile the wrapper library"""
        return self._accelerator_wrapper.git_hash()

    @property
    def accelerator_wrapper(self):
        """Return the TfliteMicroAcceleratorWrapper instance
        """
        return self._accelerator_wrapper.get_accelerator_wrapper()


    @property
    def supports_model_compilation(self) -> bool:
        """Return if this accelerator supports model compilation"""
        return type(self).compile_model != TfliteMicroAccelerator.compile_model

    
    def estimate_profiling_results(
        self, 
        results:ProfilingModelResults,
        **kwargs
    ):
        """Update the given ProfilingModelResults with estimated model metrics"""


    def enable_program_recorder(self):
        """Enable the accelerator instruction recorder"""
        return self._accelerator_wrapper.enable_program_recorder()

    def enable_data_recorder(self):
        """Enable the accelerator data recorder"""
        return self._accelerator_wrapper.enable_data_recorder()


    def compile_model(
        self, 
        model:TfliteModel,
        logger:logging.Logger=None,
        report_path:str=None,
        **kwargs
    ) -> TfliteModel:
        """Compile the given .tflite model and return a new TfliteModel instance with the compiled data
        
        NOTE: The accelerator must support model compilation to use this API
        """
        raise NotImplementedError(f'The accelerator: {self.name} does not support model compilation')