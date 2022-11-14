from __future__ import annotations
from typing import List, TypeVar, Union

import numpy as np
from . import tflite_schema as _tflite_schema_fb


TfliteModel = TypeVar('TfliteModel')




class TfliteTensor(_tflite_schema_fb.TensorT):
    """Wrapper for TFLite flatbuffer tensor"""


    def __init__(
        self, 
        index:int=-1, 
        model:TfliteModel=None, 
        fb_tensor: _tflite_schema_fb.TensorT=None
    ):
        _tflite_schema_fb.TensorT.__init__(self)
        if fb_tensor is not None:
            for x in vars(fb_tensor):
                setattr(self, x, getattr(fb_tensor, x)) 
        else:
            self.shape = None 
            self.quantization = None
        
        self._model = model
        self._index = int(index)
        self.name =  '' if not self.name else self.name.decode("utf-8")

        if model is not None and fb_tensor is not None:
            buffer = model.flatbuffer_model.buffers[fb_tensor.buffer]
            if buffer.data is not None:
                data_bytes = buffer.data.tobytes()
                a = np.frombuffer(data_bytes, dtype=self.dtype)
                if len(data_bytes) == self.shape.flat_size and len(self.shape) > 1:
                    self._data = a.reshape(self.shape)
                else:
                    self._data = a

        if not hasattr(self, '_data'):
            s = self.shape if len(self.shape) > 1 else (0,)
            self._data = np.zeros(s, dtype=self.dtype)


    @property
    def index(self) -> int:
        """Index of tensor in .tflite subgraph.tensors list"""
        return self._index 

    @property
    def dtype(self) -> np.dtype:
        """Tensor data type"""
        return tflite_to_numpy_dtype(self.type)
    @dtype.setter
    def dtype(self, v):
        self.type = numpy_to_tflite_type(v)
    @property
    def dtype_str(self) -> str:
        """Tensor data type as a string"""
        return self.dtype.__name__.replace('numpy.', '')

    @property
    def shape(self) -> TfliteShape:
        """The tensor shape"""
        return TfliteShape(self._shape)
    @shape.setter
    def shape(self, v):
        self._shape = (0, ) if v is None else v

    @property
    def quantization(self) -> TfliteQuantization:
        """Data quantization information"""
        return self._quantization 
    @quantization.setter 
    def quantization(self, v:TfliteQuantization):
        self._quantization = TfliteQuantization(v)


    @property 
    def data(self) -> np.ndarray:
        """Tensor data"""
        return self._data
    @data.setter
    def data(self, v:Union[np.ndarray,bytes]):
        """Tensor data"""
        if isinstance(v, np.ndarray):
            if v.dtype != self.dtype:
                raise ValueError(f'Data type must be {self.dtype}')
            if v.size != self.shape.flat_size:
                raise ValueError(f'Number of elements in data must be {self.shape.flat_size}')
            if len(v.shape) == 1:
                v = v.reshape(self.shape) 
            self._data = v

        else:
            self._data = np.frombuffer(v, dtype=np.uint8)

        if hasattr(self, '_model'):
            buffer = _tflite_schema_fb.BufferT()
            buffer.data = np.frombuffer(self._data.tobytes(), dtype=np.uint8)
            self._model.flatbuffer_model.buffers[self.buffer] = buffer
            self._model.regenerate_flatbuffer()
            

    @property
    def model(self) -> TfliteModel:
        """Reference to associated TfliteModel"""
        return self._model



    def shape_dtype_str(self, include_batch=False) -> str:
        """Return the shape and data-type of this tensor as a string: <dim0>x<dim1>x... (<dtype>)"""
        shape = self.shape
        if not include_batch and len(shape) > 1:
            shape = shape[1:]

        return f'{"x".join(f"{d}" for d in shape)} ({self.dtype_str})'

    def __str__(self):
        return f'{self.name}, dtype:{self.dtype_str}, shape:{self.shape}'






class TfliteShape(tuple):
    """Wrapper for tensor shape. This is a tuple of integer values"""
    def __new__ (cls, shape):
        if isinstance(shape, int):
            return super(TfliteShape, cls).__new__(cls, tuple([shape]))
        else:
            return super(TfliteShape, cls).__new__(cls, tuple([int(x) for x in shape]))
    def __str__(self):
        return 'x'.join(f'{x}' for x in self)
    
    @property
    def flat_size(self) -> int:
        """Total number of elements or flatten size"""
        n = 1 
        for x in self:
            n *= x 
        return n


class TfliteQuantization(_tflite_schema_fb.QuantizationParametersT):
    """Wrapper for tensor quantization
    
    Refer to `Quantization Specification <https://www.tensorflow.org/lite/performance/quantization_spec>`_ for more details.
    """

    def __init__(self, fb_quantization: _tflite_schema_fb.QuantizationParametersT=None):
        _tflite_schema_fb.QuantizationParametersT.__init__(self)
        if fb_quantization is not None:
            for x in vars(fb_quantization):
                setattr(self, x, getattr(fb_quantization, x))
        else:
            self.scale = [] 
            self.zeropoint = []
            self.quantization_dimension = None

    @property
    def scale(self) -> List[float]:
        """Quantization scalers as list of float values"""
        return self.__dict__.get('scale')
    @scale.setter
    def scale(self, v:List[float]):
        v = [] if v is None else [float(x) for x in v]
        self.__dict__['scale'] = v
    
    @property
    def zeropoint(self) -> List[int]:
        """Quantization zero points as list of integers"""
        return self.__dict__.get('zeroPoint')
    @zeropoint.setter
    def zeropoint(self, v):
        v = [] if v is None else [int(x) for x in v]
        self.__dict__['zeroPoint'] = v

    @property
    def quantization_dimension(self) -> int:
        """Quantization dimension"""
        return self.__dict__.get('quantizedDimension', None)
    @quantization_dimension.setter
    def quantization_dimension(self, v:int):
        self.__dict__['quantizedDimension'] = v

    @property
    def n_channels(self) -> int:
        """Number of channels. This is the number of elements in :py:attr:`~scale` and :py:attr:`~zeropoint`"""
        return len(self.scale)

    


def tflite_to_numpy_dtype(tflite_type:_tflite_schema_fb.TensorType) -> np.dtype:
    """Convert a tflite schema dtype to numpy dtype"""

    if tflite_type == _tflite_schema_fb.TensorType.FLOAT32:
        return np.float32
    elif tflite_type == _tflite_schema_fb.TensorType.FLOAT16:
        return np.float16
    elif tflite_type == _tflite_schema_fb.TensorType.INT32:
        return np.int32
    elif tflite_type == _tflite_schema_fb.TensorType.UINT8:
        return np.uint8
    elif tflite_type == _tflite_schema_fb.TensorType.INT64:
        return np.int64
    elif tflite_type == _tflite_schema_fb.TensorType.INT16:
        return np.int16
    elif tflite_type == _tflite_schema_fb.TensorType.INT8:
        return np.int8
    elif tflite_type == _tflite_schema_fb.TensorType.BOOL:
        return np.bool8
    else:
        raise ValueError(f'Unsupported .tflite tensor data type: {tflite_type}')

def numpy_to_tflite_type(dtype:np.dtype) -> _tflite_schema_fb.TensorType:
    """Convert numpy dtype to tflite schema dtype"""

    if dtype == np.float32:
        return _tflite_schema_fb.TensorType.FLOAT32
    elif dtype == np.float16: 
        return _tflite_schema_fb.TensorType.FLOAT16
    elif dtype == np.int32:
        return _tflite_schema_fb.TensorType.INT32
    elif dtype == np.uint8:
        return _tflite_schema_fb.TensorType.UINT8
    elif dtype == np.int64:
        return _tflite_schema_fb.TensorType.INT64
    elif dtype == np.int16:
        return _tflite_schema_fb.TensorType.INT16
    elif dtype == np.int8:
        return _tflite_schema_fb.TensorType.INT8
    elif dtype == np.bool8:
        return _tflite_schema_fb.TensorType.BOOL
    else:
        raise ValueError(f'Unsupported .tflite tensor data type: {dtype}')
