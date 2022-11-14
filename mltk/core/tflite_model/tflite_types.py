#pylint: disable=redefined-builtin
from __future__ import annotations
from typing import List, Tuple
from enum import IntEnum, EnumMeta
from dataclasses import dataclass, field
import numpy as np

from .tflite_tensor import TfliteTensor
from . import tflite_schema as _tflite_schema_fb



class TfliteActivation(IntEnum):
    """Activation types
    
    This may be instantiated with either an integer OR string, e.g.:

    .. highlight:: python
    .. code-block:: python

       activation = TfliteActivation(0)
       activation = TfliteActivation('relu')
    """
    NONE = _tflite_schema_fb.ActivationFunctionType.NONE
    """No activation, linear pass through"""
    RELU =  _tflite_schema_fb.ActivationFunctionType.RELU
    """Rectified Linear Unit, f(x) = max(0, x)"""
    RELU_N1_TO_1 =  _tflite_schema_fb.ActivationFunctionType.RELU_N1_TO_1
    """Rectified Linear Unit, -1 to 1, f(x) = min(max(-1, x), 1)"""
    RELU6 =  _tflite_schema_fb.ActivationFunctionType.RELU6
    """Rectified Linear Unit, 0 to 6, f(x) = min(max(0, x), 6)"""
    TANH =  _tflite_schema_fb.ActivationFunctionType.TANH
    """Hyperbolic Tangent, f(x) = tanh(x)"""
    SIGN_BIT =  _tflite_schema_fb.ActivationFunctionType.SIGN_BIT
    """Sigmoid, f(x) = e^x / (e^x + 1)"""

    def to_string(self) -> str:
        """Return the activation as a string"""
        return self.name.title()

    def __str__(self) -> str:
        return self.to_string()

# This allows for providing a string or int to the constructor
TfliteActivation.__new__ = \
    lambda cls, value: super(TfliteActivation, cls).__new__(cls, _convert_object_value_to_int(_tflite_schema_fb.ActivationFunctionType(), value))


 # This allows for providing a string or int to the constructor
 # as well as providing the width and height
class _TflitePaddingEnumMeta(EnumMeta):
    def __call__(cls, value, *args, width:int=0, height:int=0):
        self = super(EnumMeta, cls).__call__(
            _convert_object_value_to_int(_tflite_schema_fb.Padding(), value), 
        )
        if len(args) == 2:
            self.width = args[0]
            self.height = args[1]
        else:
            self.width = width 
            self.height = height 

        return self


class TflitePadding(IntEnum, metaclass=_TflitePaddingEnumMeta):
    """Padding types
    
    This may be instantiated with either an integer OR string, e.g.:

    .. highlight:: python
    .. code-block:: python

       padding = TflitePadding(0)
       padding = TflitePadding('same')
    """
    SAME = _tflite_schema_fb.Padding.SAME
    """Add zeros to the borders of the input so that the output has the same dimensions when applying the kernel
    
    .. highlight:: python
    .. code-block:: python

       out_height = ceil(float(in_height) / float(stride_height))
       out_width  = ceil(float(in_width) / float(stride_width))
    """
    VALID = _tflite_schema_fb.Padding.VALID
    """Apply the kernel only the valid regions of the input, the output dimensions will be reduced
    
    .. highlight:: python
    .. code-block:: python

       out_height = ceil(float(in_height - filter_height + 1) / float(stride_height))
       out_width  = ceil(float(in_width - filter_width + 1) / float(stride_width))

    """

    @property
    def width(self) -> int:
        """The calculated width of the padding for the current layer"""
        return getattr(self, '_width', 0)
    @width.setter
    def width(self, v:int):
        setattr(self, '_width', v)

    @property
    def height(self) -> int:
        """The calculated height of the padding for the current layer"""
        return getattr(self, '_height', 0)
    @height.setter
    def height(self, v:int):
        setattr(self, '_height', v)

    def to_string(self) -> str:
        """Return the padding as a string"""
        return self.name.title()

    def __str__(self) -> str:
        return self.to_string()



@dataclass
class TfliteConvParams:
    """Calculated Convolution Parameters
    
    .. seealso::
    
       - `ConvParamsQuantize <https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/kernels/conv_common.cc#L54>`_
       - `Quantization Specification <https://www.tensorflow.org/lite/performance/quantization_spec>`_
    """
    padding:TflitePadding = field(default_factory=lambda: TflitePadding.SAME)
    """Kernel padding"""
    stride_width:int = 0
    """Kernel width"""
    stride_height:int = 0
    """Kernel height"""
    dilation_width_factor:int = 1
    """Kernel dilation width factor"""
    dilation_height_factor:int = 1
    """Kernel dilation height factor"""
    input_offset:int = 0
    """Input quantization offset (i.e. zero point)"""
    weights_offset:int = 0
    """Weight (aka filters) quantization offset (i.e. zero point)"""
    output_offset:int = 0
    """Output quantization offset (i.e. zero point)"""
    per_channel_output_multiplier:List[int] = field(default_factory=list)
    """Per layer multipliers for output scalers"""
    per_channel_output_shift:List[int] = field(default_factory=list)
    """Per layer shifts for output scalers"""
    quantized_activation_min:int = 0
    """Fused activation min value"""
    quantized_activation_max:int = 0
    """Fused activation max value"""


    @staticmethod
    def calculate(layer) -> TfliteConvParams:
        """Calculate the parameters for the given layer"""
        from .tflite_layer import TfliteConv2dLayer
        layer:TfliteConv2dLayer = layer
        options = layer.options

        input = layer.input_tensor
        filters = layer.filters_tensor
        output = layer.output_tensor

        input_height = input.shape[1]
        input_width = input.shape[2]
        filter_height = filters.shape[1]
        filter_width = filters.shape[2]

        params = TfliteConvParams()
        params.padding = options.padding
        params.stride_width = options.stride_width
        params.stride_height = options.stride_height
        params.dilation_width_factor = options.dilationWFactor
        params.dilation_height_factor = options.dilationHFactor
        params.input_offset = int(-input.quantization.zeropoint[0])
        params.weights_offset = int(-filters.quantization.zeropoint[0])
        params.output_offset = int(output.quantization.zeropoint[0])
        

        _compute_padding_height_width(
            padding = params.padding,
            stride_height = options.stride_height,
            stride_width = options.stride_width,
            dilation_height_factor = options.dilationHFactor,
            dilation_width_factor = options.dilationWFactor,
            height = input_height,
            width = input_width,
            filter_height = filter_height,
            filter_width = filter_width
        )
        _populate_convolution_quantization_params(
            input=input,
            filters=filters,
            output=output,
            per_channel_multiplier=params.per_channel_output_multiplier,
            per_channel_shift=params.per_channel_output_shift,
        )

        params.quantized_activation_min, params.quantized_activation_max = \
            _calculate_activation_range_quantized(
                activation=options.activation,
                output=output,
            )

        return params


@dataclass
class TfliteDepthwiseConvParams:
    """Calculated Depthwise Convolution Parameters
    
    .. seealso::
    
       - `DepthwiseConvParamsQuantized <https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/kernels/depthwise_conv_common.cc#L59>`_
       - `Quantization Specification <https://www.tensorflow.org/lite/performance/quantization_spec>`_
    """
    padding:TflitePadding = field(default_factory=lambda: TflitePadding.SAME)
    """Kernel padding"""
    stride_width:int = 0
    """Kernel width"""
    stride_height:int = 0
    """Kernel height"""
    dilation_width_factor:int = 1
    """Kernel dilation width factor"""
    dilation_height_factor:int = 1
    """Kernel dilation height factor"""
    depth_multiplier:int = 0
    """Depth multiplier"""
    input_offset:int = 0
    """Input quantization offset (i.e. zero point)"""
    weights_offset:int = 0
    """Weight (aka filters) quantization offset (i.e. zero point)"""
    output_offset:int = 0
    """Output quantization offset (i.e. zero point)"""
    per_channel_output_multiplier:List[int] = field(default_factory=list)
    """Per layer multipliers for output scalers"""
    per_channel_output_shift:List[int] = field(default_factory=list)
    """Per layer shifts for output scalers"""
    quantized_activation_min:int = 0
    """Fused activation min value"""
    quantized_activation_max:int = 0
    """Fused activation max value"""


    @staticmethod
    def calculate(layer) -> TfliteDepthwiseConvParams:
        """Calculate the parameters for the given layer"""
        from .tflite_layer import TfliteDepthwiseConv2dLayer
        layer:TfliteDepthwiseConv2dLayer = layer
        options = layer.options

        input = layer.input_tensor
        filters = layer.filters_tensor
        output = layer.output_tensor

        input_height = input.shape[1]
        input_width = input.shape[2]
        filter_height = filters.shape[1]
        filter_width = filters.shape[2]

        params = TfliteDepthwiseConvParams()
        params.padding = options.padding
        params.stride_width = options.stride_width
        params.stride_height = options.stride_height
        params.dilation_width_factor = options.dilationWFactor
        params.dilation_height_factor = options.dilationHFactor
        params.depth_multiplier = options.depthMultiplier
        params.input_offset = int(-input.quantization.zeropoint[0])
        params.weights_offset = int(-filters.quantization.zeropoint[0])
        params.output_offset = int(output.quantization.zeropoint[0])
        

        _compute_padding_height_width(
            padding = params.padding,
            stride_height = options.stride_height,
            stride_width = options.stride_width,
            dilation_height_factor = options.dilationHFactor,
            dilation_width_factor = options.dilationWFactor,
            height = input_height,
            width = input_width,
            filter_height = filter_height,
            filter_width = filter_width
        )
        _populate_convolution_quantization_params(
            input=input,
            filters=filters,
            output=output,
            per_channel_multiplier=params.per_channel_output_multiplier,
            per_channel_shift=params.per_channel_output_shift,
        )

        params.quantized_activation_min, params.quantized_activation_max = \
            _calculate_activation_range_quantized(
                activation=options.activation,
                output=output,
            )

        return params


@dataclass
class TfliteFullyConnectedParams:
    """Calculated Full Connected Parameters
    
    .. seealso::
    
       - `FullyConnectedParamsQuantized <https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/kernels/fully_connected_common.cc#L34>`_
       - `Quantization Specification <https://www.tensorflow.org/lite/performance/quantization_spec>`_
    """

    input_offset:int = 0
    """Input quantization offset (i.e. zero point)"""
    weights_offset:int = 0
    """Weight (aka filters) quantization offset (i.e. zero point)"""
    output_offset:int = 0
    """Output quantization offset (i.e. zero point)"""
    output_multiplier:int = 0
    """Output multiplier for its quantization scaler"""
    output_shift:int = 0
    """Output shift for its quantization scaler"""
    quantized_activation_min:int = 0
    """Fused activation min value"""
    quantized_activation_max:int = 0
    """Fused activation max value"""


    @staticmethod
    def calculate(layer) -> TfliteFullyConnectedParams:
        """Calculate the parameters for the given layer"""
        from .tflite_layer import TfliteFullyConnectedLayer
        layer:TfliteFullyConnectedLayer = layer
        options = layer.options

        input = layer.input_tensor
        weights = layer.weights_tensor
        output = layer.output_tensor

        params = TfliteFullyConnectedParams()
        params.input_offset = int(-input.quantization.zeropoint[0])
        params.weights_offset = int(-weights.quantization.zeropoint[0])
        params.output_offset = int(output.quantization.zeropoint[0])
        
        input_product_scale = float(np.float32(input.quantization.scale[0]) * np.float32(weights.quantization.scale[0])) # NOTE: TFLM uses float32 for the multiplication
        real_multiplier = input_product_scale / output.quantization.scale[0]
        params.output_multiplier, params.output_shift = _quantize_multiplier(real_multiplier)

        params.quantized_activation_min, params.quantized_activation_max = \
            _calculate_activation_range_quantized(
                activation=options.activation,
                output=output,
            )

        return params


@dataclass
class TflitePoolParams:
    """Calculated Pooling Parameters
    
    .. seealso::
    
       - `AveragePoolingEvalQuantized <https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/kernels/pooling_common.cc#L105>`_
       - `Quantization Specification <https://www.tensorflow.org/lite/performance/quantization_spec>`_
    """
    padding:TflitePadding = field(default_factory=lambda: TflitePadding.SAME)
    """Kernel padding"""
    stride_width:int = 0
    """Kernel width"""
    stride_height:int = 0
    """Kernel height"""
    quantized_activation_min:int = 0
    """Fused activation min value"""
    quantized_activation_max:int = 0
    """Fused activation max value"""


    @staticmethod
    def calculate(layer) -> TflitePoolParams:
        """Calculate the parameters for the given layer"""
        from .tflite_layer import TflitePooling2dLayer
        layer:TflitePooling2dLayer = layer
        options = layer.options

        input = layer.input_tensor
        output = layer.output_tensor

        input_height = input.shape[1]
        input_width = input.shape[2]
        filter_height = options.filter_height
        filter_width = options.filter_width

        params = TflitePoolParams()
        params.padding = options.padding
        params.stride_width = options.stride_width
        params.stride_height = options.stride_height

        _compute_padding_height_width(
            padding = params.padding,
            stride_height = options.stride_height,
            stride_width = options.stride_width,
            dilation_height_factor = 1,
            dilation_width_factor = 1,
            height = input_height,
            width = input_width,
            filter_height = filter_height,
            filter_width = filter_width
        )

        params.quantized_activation_min, params.quantized_activation_max = \
            _calculate_activation_range_quantized(
                activation=options.activation,
                output=output,
            )

        return params



def _compute_padding_height_width(
    padding: TflitePadding,
    stride_height:int,
    stride_width:int,
    dilation_height_factor:int, 
    dilation_width_factor:int,
    height:int,
    width:int,
    filter_height:int,
    filter_width:int,
):
    out_width = _compute_out_size(
        padding_type=padding,
        image_size=width,
        filter_size=filter_width,
        stride=stride_width,
        dilation_rate=dilation_width_factor
    )

    out_height = _compute_out_size(
        padding_type=padding,
        image_size=height,
        filter_size=filter_height,
        stride=stride_height,
        dilation_rate=dilation_height_factor
    )

    padding.width, _ = _compute_padding_with_offset(
        stride=stride_width,
        dilation_rate=dilation_width_factor,
        in_size=width,
        filter_size=filter_width,
        out_size=out_width
    )

    padding.height, _ = _compute_padding_with_offset(
        stride=stride_height,
        dilation_rate=dilation_height_factor,
        in_size=height,
        filter_size=filter_height,
        out_size=out_height
    )


def _populate_convolution_quantization_params(
    input: TfliteTensor,
    filters: TfliteTensor,
    output: TfliteTensor,
    per_channel_multiplier:List[int],
    per_channel_shift:List[int],
) -> Tuple[int, int]:
    num_channels = filters.shape[filters.quantization.quantization_dimension]
    
    input_scale = input.quantization.scale[0]
    output_scale = output.quantization.scale[0]
    filter_scales = filters.quantization.scale
    is_per_channel = len(filter_scales) > 1

    for i in range(num_channels):
        # If per-tensor quantization parameter is specified, broadcast it along the
        # quantization dimension (channels_out).
        filter_scale = filter_scales[i] if is_per_channel else filter_scales[0]
        effective_output_scale = input_scale * filter_scale / output_scale

        significand, channel_shift = _quantize_multiplier(effective_output_scale)
        per_channel_multiplier.append(significand)
        per_channel_shift.append(channel_shift)
  

def _calculate_activation_range_quantized(
    activation:TfliteActivation,
    output:TfliteTensor
) -> Tuple[int, int]:
    scale = output.quantization.scale[0]
    zeropoint = output.quantization.zeropoint

    qmin = np.iinfo(output.dtype).min
    qmax = np.iinfo(output.dtype).max

    if activation == TfliteActivation.RELU:
        tmp_q = _quantize(scale, zeropoint, 0.0)
        act_min = max(qmin, tmp_q)
        act_max = qmax
    
    elif activation == TfliteActivation.RELU6:
        tmp_q = _quantize(scale, zeropoint, 0.0)
        act_min = max(qmin, tmp_q)

        tmp_q = _quantize(scale, zeropoint, 6.0)
        act_max = min(qmax, tmp_q)

    elif activation == TfliteActivation.RELU_N1_TO_1:
        tmp_q = _quantize(scale, zeropoint, -1.0)
        act_min = max(qmin, tmp_q)

        tmp_q = _quantize(scale, zeropoint, 1.0)
        act_max = min(qmax, tmp_q)

    else:
        act_min = qmin 
        act_max = qmax

    return act_min, act_max



def _compute_out_size(
    padding_type: TflitePadding,
    image_size:int,
    filter_size:int,
    stride:int,
    dilation_rate:int = 1,
) -> int:
    effective_filter_size = (filter_size - 1) * dilation_rate + 1
    if stride == 0:
        return 0

    if padding_type == TflitePadding.SAME:
        return (image_size + stride - 1) // stride
    elif padding_type == TflitePadding.VALID:
        return (image_size + stride - effective_filter_size) // stride
    else:
        return 0


def _compute_padding_with_offset(
    stride:int,
    dilation_rate:int,
    in_size:int,
    filter_size:int,
    out_size:int,
) -> Tuple[int, int]:
    effective_filter_size = (filter_size - 1) * dilation_rate + 1
    total_padding = ((out_size - 1) * stride + effective_filter_size - in_size)
    total_padding = max(total_padding, 0)

    padding = total_padding // 2
    offset = total_padding % 2

    return padding, offset


def _quantize_multiplier(double_multiplier:float) -> Tuple[int, int]:
    '''
        Quantize floating point multiplier: returns 'shift' and 'quantized_multiplier'
        such that double_multiplier = quantized_multiplier * (2**shift), where shift
        is a signed integer and quantized_multiplier is q(1.31) in [0.5, 1.0)
    '''
    if double_multiplier == 0:
        quantized_multiplier=0
        shift=0
    else:
        shift = np.ceil(np.log2(double_multiplier))
        quantized_multiplier = (2**-shift)*double_multiplier
        quantized_multiplier = (2**32)*quantized_multiplier
        quantized_multiplier += 1
        quantized_multiplier /= 2

    return int(quantized_multiplier), shift


_int32_max = np.iinfo(np.int32).max
_int32_min = np.iinfo(np.int32).min

def _quantize(scale:float, zeropoint:int, f:float) -> int:
    tmp = round(f / scale)
    assert tmp >= _int32_min and tmp <= _int32_max
    q = zeropoint + int(tmp)
    return q
    

def _convert_object_value_to_int(obj:object, needle:str, default:int=-1) -> int:
    if isinstance(needle, int):
        return needle
    
    for key in dir(obj):
        if key.lower() == needle.lower():
            return getattr(obj, key)

    return default

