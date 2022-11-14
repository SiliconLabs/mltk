from typing import List, Dict
import re
import collections
import numpy as np
import msgpack


from mltk.utils.string_formatting import  format_units

from .tflite_micro_accelerator import TfliteMicroAccelerator


class TfliteMicroModelDetails:
    """TF-Lite Micro Model Details"""

    def __init__(self, wrapper_details:dict):
        self._details:dict = wrapper_details
    
    @property 
    def name(self) -> str:
        """Name of model"""
        return self._details['name']
    @property 
    def version(self)-> int:
        """Version of model"""
        return self._details['version']
    @property 
    def date(self)-> str:
        """Date of model in ISO8601 format"""
        return self._details['date']
    @property 
    def description(self)-> str:
        """Description of model"""
        return self._details['description']
    @property 
    def classes(self) -> List[str]:
        """List of class labels"""
        return self._details['classes']
    @property 
    def hash(self)-> str:
        """Unique hash of model data"""
        return self._details['hash']
    @property 
    def accelerator(self)-> str:
        """Accelerater kernels loaded into TFLM interpreter"""
        return self._details['accelerator']
    @property 
    def runtime_memory_size(self)-> int:
        """Total amount of RAM required at runtime to run model"""
        return self._details['runtime_memory_size']


    def __str__(self):
        s = ''
        s += f"Name: {self.name}\n"
        s += f"Version: {self.version}\n"
        s += f"Date: {self.date}\n"
        s += f"Description: {self.description}\n"
        s += f"Hash: {self.hash}\n"
        s += f"Accelerator: {self.accelerator}\n"
        s += f"Classes: {', '.join(self.classes)}\n"
        s += f"Total runtime memory: {format_units(self.runtime_memory_size)}Bytes\n"
        return s


class TfliteMicroLayerError:
    WARNING_RE = re.compile(r'.*Op(\d+)-(\S+)\ not supported: (.*)')

    def __init__(self, index:int, name:str, msg:str):
        self._index = index 
        self._name = name
        self._msg = msg 

    @property
    def index(self) -> int:
        """The layer's index"""
        return self._index
    @property
    def name(self) -> int:
        """The layer's full name"""
        return self._name
    @property
    def msg(self) -> int:
        """The layer's error msg"""
        return self._msg

    @staticmethod
    def _parse_error_log(msg):
        match = TfliteMicroLayerError.WARNING_RE.match(msg)
        if not match:
            return None 
        return TfliteMicroLayerError( 
            index=int(match.group(1)),
            name=msg.split()[0],
            msg=match.group(3)
        )

    def __str__(self) -> str:
        return f'{self.name}: {self.msg}'


class TfliteMicroProfiledLayerResult(collections.defaultdict):
    """Result of profiling a specifc model layer"""
    def __init__(self, result:dict):
        collections.defaultdict.__init__(self, lambda: 0, result)

    @property
    def name(self) -> int:
        """Name of layer"""
        return self['name']
    @property
    def macs(self) -> int:
        """Number of Multiple-Accumulate operations required by this profiler"""
        return self['macs']
    @property
    def ops(self) -> int:
        """Number of operations required by this profiler"""
        return self['ops']
    @property
    def accelerator_cycles(self) -> int:
        """Number of accelerator clock cycles required by this profiler"""
        return self['accelerator_cycles']
    @property
    def time(self) -> float:
        """Time in seconds required by this layer"""
        return self['time']
    @property
    def cpu_cycles(self) -> float:
        """Number of CPU clock cycles required by this layer"""
        return self['cpu_cycles']
    @property
    def energy(self) -> float:
        """Energy in Joules required by this layer
        The energy is relative to the 'baseline' energy (i.e. energy used while the device was idling)"""
        return self['energy']



class TfliteMicroModel:
    """This class wrappers the TF-Lite Micro interpreter loaded with a .tflite model"""

    def __init__(
        self, 
        tflm_wrapper, 
        tflm_accelerator:TfliteMicroAccelerator, 
        flatbuffer_data:bytes, 
        enable_profiler:bool=False,
        enable_tensor_recorder:bool=False,
        force_buffer_overlap:bool=False,
        runtime_buffer_size:int=0,
    ):
        # pylint: disable=protected-access
        from .tflite_micro import TfliteMicro

        TfliteMicro._clear_logged_errors()
        accelerator_wrapper = None if tflm_accelerator is None else tflm_accelerator.accelerator_wrapper
        self._model_wrapper = tflm_wrapper.TfliteMicroModelWrapper()
        if not self._model_wrapper.load(
            flatbuffer_data,
            accelerator_wrapper, 
            enable_profiler,
            enable_tensor_recorder,
            force_buffer_overlap,
            runtime_buffer_size
        ):
            raise Exception(
                f'Failed to load model, additional info:\n{TfliteMicro._get_logged_errors_str()}'
            )

        self._tflm_accelerator = tflm_accelerator
        self._layer_errors:List[TfliteMicroLayerError] = []
        for msg in TfliteMicro._get_logged_errors():
            err = TfliteMicroLayerError._parse_error_log(msg)
            if err:
                self._layer_errors.append(err)  

    @property
    def accelerator(self) -> TfliteMicroAccelerator:
        """Reference to hardware accelerator used by model"""
        return self._tflm_accelerator

    @property
    def layer_errors(self) -> List[TfliteMicroLayerError]:
        """List of error messages triggered by kernels while loading the model.
        Typically, these errors indicate that a given model layer is not supported by
        a hardware accelerator and had to fallback to a default kernel implementation.
        """
        return self._layer_errors

    @property
    def details(self) -> TfliteMicroModelDetails:
        """Return details about loaded model"""
        return TfliteMicroModelDetails(self._model_wrapper.get_details())
        
    @property
    def input_size(self) -> int:
        """Number of input tensors"""
        return self._model_wrapper.get_input_size()


    def input(self, index = 0, value: np.ndarray=None) -> np.ndarray:
        """Return a reference to a model input tensor's data
        If the value argument is provided then copy the value to the input tensor's buffer
        """
        if index >= self.input_size:
            raise IndexError(f'Input index: {index} >= max size: { self.input_size}')
        input_tensor = self._model_wrapper.get_input(index)
        if value is not None:
            np.copyto(input_tensor, value)

        return input_tensor

    @property
    def output_size(self) -> int:
        """Number of output tensors"""
        return self._model_wrapper.get_output_size()


    def output(self, index = 0) -> np.ndarray:
        """Return a reference to a model output tensor's data
        """
        if index >= self.output_size:
            raise IndexError(f'Output index: {index} >= max size: { self.output_size}')
        return self._model_wrapper.get_output(index)


    def invoke(self):
        """Invoke the model to execute one inference"""
        # pylint: disable=protected-access
        from .tflite_micro import TfliteMicro

        TfliteMicro._clear_logged_errors() 
        if not self._model_wrapper.invoke():
            raise Exception(f'Failed to invoke model, additional info:\n{TfliteMicro._get_logged_errors_str()}')


    @property
    def is_profiler_enabled(self) -> bool:
        """Return if the profiler is enabled"""
        return self._model_wrapper.is_profiler_enabled()


    def get_profiling_results(self) -> List[TfliteMicroProfiledLayerResult]:
        """Return the profiling results of each model layer
        
        Returns:
            A list where each entry contains the profiling results
            of the associated model layer
        """
        retval = []
        results = self._model_wrapper.get_profiling_results()
        for e in results:
            retval.append(TfliteMicroProfiledLayerResult(e))
        return retval


    @property
    def is_tensor_recorder_enabled(self) -> bool:
        """Return if the tensor recorder is enabled """
        return self._model_wrapper.is_tensor_recorder_enabled()


    def get_recorded_data(self) -> List[Dict[str,object]]:
        """Return the recorded contents of each model layers' tensors
        
        Returns:
            A list where each entry contains the input/output tensors
            of the associated model layer
        """
        results_bin = self._model_wrapper.get_recorded_data()
        if results_bin is None:
            raise RuntimeError('Failed to retrieve recorded model data from Tensorflow-Lite Micro')

        try:
            recorded_data = msgpack.loads(results_bin)
        except Exception as e:
            raise RuntimeError(f'Failed to parse recorded model binary data, err: {e}')

        return recorded_data

    def get_layer_error(self, index:int) -> TfliteMicroLayerError:
        """Return the TfliteMicroLayerError at the given layer index if found else return None"""
        for err in self._layer_errors:
            if err.index == index:
                return err 
        return None

    
    def __str__(self) -> str:
        return f'{self.details}'