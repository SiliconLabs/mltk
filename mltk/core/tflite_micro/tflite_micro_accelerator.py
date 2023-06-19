from typing import List
import logging
from mltk.core.profiling_results import ProfilingModelResults
from mltk.core.tflite_model import TfliteModel
from mltk.utils.python import get_case_insensitive


class TfliteMicroAccelerator:
    """TF-Lite Micro Accelerator

    This class allows for providing hardware-accelerated
    kernels to the TFLM interpreter.
    """

    def __init__(self, accelerator_wrapper):
        self._accelerator_wrapper = accelerator_wrapper
        self._active_variant:str = None

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        result._accelerator_wrapper = self._accelerator_wrapper
        result._active_variant = self._active_variant
        memo[id(self)] = result
        return result

    @property
    def name(self) -> str:
        """The name of the accelerator"""
        return self._accelerator_wrapper.name()

    @property
    def variants(self) -> List[str]:
        """List of variants supported by this accelerator"""
        if hasattr(self._accelerator_wrapper, 'variants'):
            return self._accelerator_wrapper.variants()
        else:
            return [self.name]

    @property
    def active_variant(self) -> str:
        """The name of the variant actively being used by this accelerator"""
        return self._active_variant
    @active_variant.setter
    def active_variant(self, v:str):
        v = get_case_insensitive(v, self.variants)
        if not v:
            raise ValueError('Unknown variant')
        self._active_variant = v

    @property
    def api_version(self) -> int:
        """The API version number this wrapper was built with
        This number must match the tflite_micro_wrapper's API version
        """
        if hasattr(self._accelerator_wrapper, 'api_version'):
            return self._accelerator_wrapper.api_version()
        else:
            return None

    @property
    def git_hash(self) -> str:
        """Return the GIT hash of the MLTK repo used to compile the wrapper library"""
        if hasattr(self._accelerator_wrapper, 'git_hash'):
            return self._accelerator_wrapper.git_hash()
        else:
            return None

    @property
    def accelerator_wrapper(self) -> object:
        """Return the TfliteMicroAcceleratorWrapper instance
        """
        if hasattr(self._accelerator_wrapper, 'get_accelerator_wrapper'):
            return self._accelerator_wrapper.get_accelerator_wrapper()
        else:
            return None


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
        if hasattr(self._accelerator_wrapper, 'enable_program_recorder'):
            return self._accelerator_wrapper.enable_program_recorder()
        else:
            return None

    def enable_data_recorder(self):
        """Enable the accelerator data recorder"""
        if hasattr(self._accelerator_wrapper, 'enable_data_recorder'):
            return self._accelerator_wrapper.enable_data_recorder()
        else:
            return None


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
        raise RuntimeError(f'The accelerator: {self.name} does not support model compilation')



class PlaceholderTfliteMicroAccelerator(TfliteMicroAccelerator):
    """This accelerator does NOT have a corresponding Python wrapper"""
    def __init__(self, name:str):
        self._name = name
        super().__init__(None)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        result._accelerator_wrapper = None
        result._name = self._name
        result._active_variant = self._active_variant
        memo[id(self)] = result
        return result

    @property
    def name(self) -> str:
        return self._name

    @property
    def api_version(self) -> int:
        from mltk.core.tflite_micro import TfliteMicro
        return TfliteMicro.api_version()