import importlib
from mltk.core.profiling_results import ProfilingModelResults
from mltk.core.tflite_micro.tflite_micro_accelerator import TfliteMicroAccelerator
from mltk.core.utils import get_mltk_logger


class MVPTfliteMicroAccelerator(TfliteMicroAccelerator):
    """This class allows for running .tflite models on the MVP hardware simulator"""
    def __init__(self):
        # Import the MVP C++ python wrapper
        # For more details, see:
        # <mltk root>/cpp/mvp_wrapper
        try:
            _wrapper = importlib.import_module('mltk.core.tflite_micro.accelerators.mvp._mvp_wrapper')
        except (ImportError, ModuleNotFoundError) as e:
            raise ImportError(f'Failed to import the MVP wrapper C++ shared library, err: {e}\n' \
                            'This likely means you need to re-build the mltk package or MVP wrapper package\n\n') from e

        super().__init__(_wrapper)

    def set_simulator_backend_enabled(self, enabled:bool):
        """Enable/disable the simulator backend. This is used internally"""
        self._accelerator_wrapper.set_simulator_backend_enabled(enabled)
    
    def set_calculate_accelerator_cycles_only_enabled(self, enabled:bool):
        """Enable/disabe only calculating accelerator cycles during simulation. This is used internally"""
        self._accelerator_wrapper.set_calculate_accelerator_cycles_only_enabled(enabled)


    def estimate_profiling_results(
        self, 
        results:ProfilingModelResults,
        **kwargs
    ):
        """Update the given ProfilingModelResults with estimated model metrics"""
        from .estimator import get_estimates

        # If not clock rate was given, then just default to 78MHz
        # as that's the max rate that can be used with the radio
        if results.cpu_clock_rate == 0:
            # pylint: disable=protected-access
            results._cpu_clock_rate = int(78e6)

        for layer in results.layers:
            try:
                get_estimates(
                    layer=layer, 
                    accelerator=results.accelerator,
                    cpu_clock_rate=results.cpu_clock_rate,
                )
            except Exception as e:
                get_mltk_logger().warning(f'Failed to get profiling estimates for layer: {layer.name}, err: {e}')
