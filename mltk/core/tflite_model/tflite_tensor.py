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
                data_bytes = buffer.data.tobytes() if isinstance(buffer.data, np.ndarray) else bytes(buffer.data)

                if  hasattr(_tflite_schema_fb.TensorType, 'INT4') and self.type == _tflite_schema_fb.TensorType.INT4:
                    # NumPy does not support int4 so we have to expand to int8
                    # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/portable_tensor_utils.cc
                    # UnpackDenseInt4IntoInt8()
                    n_elements = self.shape.flat_size
                    raw_data_array = np.empty((n_elements,), dtype=np.int8)

                    sign_bit_mask = 1 << (4 - 1)
                    def sign_extend_4bits(value):
                        return (value & (sign_bit_mask-1)) - (value & sign_bit_mask)

                    for i in range(n_elements // 2):
                        v = data_bytes[i]
                        lower = sign_extend_4bits(v & 0x0F)
                        upper = sign_extend_4bits((v & 0xF0) >> 4)
                        raw_data_array[i*2 + 0] = lower
                        raw_data_array[i*2 + 1] = upper

                    # If the buffer size is odd, extract the final lower nibble.
                    if n_elements % 2 != 0:
                        v = data_bytes[n_elements//2]
                        lower = sign_extend_4bits((v & 0xF0) >> 4)
                        raw_data_array[-1] = lower

                else:
                    raw_data_array = np.frombuffer(data_bytes, dtype=self.dtype)

                if raw_data_array.size == self.shape.flat_size and len(self.shape) > 1:
                    self._data = raw_data_array.reshape(self.shape)
                else:
                    self._data = raw_data_array

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
    def is_variable(self) ->bool:
        """True if this tensor is populated at runtime and its state persists between inferences"""
        return self.isVariable

    @property
    def size_bytes(self) -> int:
        """The number of bytes required to hold the data for this tensor"""
        if self._data is None:
            return 0
        return self.data.nbytes

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
            data_bytes = self._data.tobytes()

            if hasattr(_tflite_schema_fb.TensorType, 'INT4') and self.type == _tflite_schema_fb.TensorType.INT4 and isinstance(v, np.ndarray):
                # NumPy does not support int4 so we have to pack the two int8 values into 1 byte
                # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/portable_tensor_utils.cc

                data_bytes_len = len(data_bytes)
                packed_data = bytearray()

                for i in range(data_bytes_len//2):
                    lower = data_bytes[i*2 + 0]
                    upper = data_bytes[i*2 + 1]
                    assert lower.bit_length() <= 4
                    assert upper.bit_length() <= 4
                    packed_value = (lower & 0x0F) | ((upper & 0x0F) << 4)
                    packed_data.append(packed_value)

                if data_bytes_len % 2 != 0:
                    lower = data_bytes[-1]
                    assert lower.bit_length() <= 4
                    packed_value = (lower & 0x0F)
                    packed_data.append(packed_value)

                data_bytes = packed_data

            buffer.data = np.frombuffer(data_bytes, dtype=np.uint8)

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
    elif hasattr(_tflite_schema_fb.TensorType, 'INT4') and tflite_type == _tflite_schema_fb.TensorType.INT4:
        # Numpy does not support 4-bit, so we have to use int8
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
