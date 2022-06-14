from typing import Tuple, Union, List
import math
import numpy as np


from mltk.utils.python import as_list
from ... import (
    TfliteShape,
    TfliteTensor,
)

kAsymmetricInt8Max = 127
kAsymmetricInt8Min = -128
kSymmetricInt8Scale = kAsymmetricInt8Max


class TfliteLayerGeneratorConfig(dict):

    @property
    def activation(self) -> str:
        entry = self.get_config_entry('activation', throw_exception=True)
        if entry is None:
            entry = 'none'
        entry = entry.lower()
        if entry not in ('none', 'relu', 'relu6'):
            raise RuntimeError('Invalid activation')

        return entry

    @property
    def padding(self) -> str:
        entry = self.get_config_entry('padding', throw_exception=True)
        entry = entry.lower()
        if entry not in ('same', 'valid'):
            raise RuntimeError('Padding must be "same" or "valid"')
        return entry 

    @property
    def stride(self) -> Tuple[int, int]:
        return self.get_shape_config_entry('stride', 2, throw_exception=True)

    @property
    def filter(self) -> Tuple[int, int]:
        return self.get_shape_config_entry('filter', 2, throw_exception=True)


    def get_config_entry(self, key:str, throw_exception=False):
        if key not in self:
            if throw_exception:
                raise RuntimeError(f'Layer config missing entry: {key}')
            return None 
        return self.get(key)

    def get_shape_config_entry(self, key:str, size:int, throw_exception=False):
        entry = self.get_config_entry(key, throw_exception)
        if entry is None:
            return None 
        
        if isinstance(entry, str):
            try:
                entry = [int(x) for x in entry.split('x')]
            except:
                raise RuntimeError(f'Failed to parse "{key}" config entry, must be a shape')
        elif not isinstance(entry, (list,tuple)):
            raise RuntimeError(f'Failed to parse "{key}" config entry, must be a shape')
        
        if len(entry) != size:
            raise RuntimeError(f'Config entry {key} must have {size} dimensions')

        return entry

    def get_range_config_entry(self, key:str):
        entry = self.get_config_entry(key)
        if entry is None:
            return None 
        
        try:
            retval = [float(x.strip()) for x in entry.split(',')]
            if len(retval) != 2:
                raise Exception
        except:
            raise RuntimeError(f'Failed to parse "{key}" config entry, must be of the form: <low>,<high>')

        return retval


class TfliteTensorGenerator(TfliteTensor):

    def __init__(
        self, 
        data:np.ndarray,
        shape:Tuple=None,
        scale:Union[float, List[float]] = None,
        zeropoint:Union[int, List[int]] = None,
        qdim:int = 0,
        is_model_input=False
    ):
        TfliteTensor.__init__(self)
        self.shape = TfliteShape(shape or data.shape)
        self.name = ''
        self.dtype = np.int8 if data is None else data.dtype 
        self._data = data 
        self.quantization.scale = as_list(scale)
        self.quantization.zeropoint = as_list(zeropoint)
        self.quantization.quantization_dimension = qdim
        self.is_model_input = is_model_input




def generate_asymmetric_quantized_tensor(
    config:TfliteLayerGeneratorConfig, 
    name:str, 
    generate_random=True,
    shape:Tuple=None,
    default_scale:float=None, 
    default_zeropoint:int=None,
    is_model_input=False
) -> TfliteTensorGenerator:
    shape = shape or config.get_shape_config_entry( name, 4, throw_exception=True)
    drange = config.get_range_config_entry(f'{name}_range') or config.get_range_config_entry( f'range')
    data = config.get_config_entry(f'{name}_data')
    scale = config.get_config_entry(f'{name}_scale')
    zeropoint = config.get_config_entry(f'{name}_zeropoint')

    data = _generate_tensor_data(
        shape, 
        data, 
        drange, 
        generate_random=generate_random,
    )

    qdata = None
    if data is not None:
        if scale is None or zeropoint is None:
            min_value = 0
            max_value = 0
            for x in data.flatten():
                min_value = min(min_value, x)
                max_value = max(max_value, x)

            if scale is None:
                scale = (max_value - min_value) / (128 - (-127))
            if zeropoint is None:
                zeropoint = -127 + math.floor(-min_value / scale + .5)

        qdata = np.empty(data.shape, np.int8)
        qdata_flat = qdata.reshape(-1)
        data_flat = data.reshape(-1)
        for i in range(len(data_flat)): #pylint: disable=consider-using-enumerate
            qdata_flat[i] = _asymmetric_quantize_float_to_int8(data_flat[i], scale, zeropoint)

    scale = scale or default_scale
    zeropoint = zeropoint or default_zeropoint

    return TfliteTensorGenerator(
        qdata,
        shape=shape,
        scale=scale, 
        zeropoint=zeropoint,
        is_model_input=is_model_input
    )


def generate_signed_symmetric_quantized_tensor(
    config:TfliteLayerGeneratorConfig, 
    name:str, 
    generate_random=True,
    shape:Tuple=None,
    scale:float=None, 
    is_model_input=False,
    dtype:np.dtype=None,
    qdrange:Tuple[int,int]=None,
    sparsity:float=None
) -> TfliteTensorGenerator:
    shape = shape or config.get_shape_config_entry(name, 4, throw_exception=True)
    drange = config.get_range_config_entry(f'{name}_range') or config.get_range_config_entry(f'range')
    data = config.get_config_entry(f'{name}_data')
    qdrange = qdrange or (kAsymmetricInt8Min, kAsymmetricInt8Max)

    data = _generate_tensor_data(
        shape, 
        data, 
        drange, 
        generate_random=generate_random,
    )

    qdata = None
    if data is not None:
        if scale is None:
            min_value = 0
            max_value = 0
            for x in data.flatten():
                min_value = min(min_value, x)
                max_value = max(max_value, x)

            scale = (max_value - min_value) / (qdrange[1] - qdrange[0])

        qdata = np.empty(data.shape, dtype=dtype or np.int32)
        qdata_flat = qdata.reshape(-1)
        data_flat = data.reshape(-1)
        for i in range(len(data_flat)): #pylint: disable=consider-using-enumerate
            quantized_value =  _symmetric_quantize_float_to_int32(data_flat[i], scale)
            qdata_flat[i] = min(qdrange[1], max(qdrange[0], quantized_value))

    return TfliteTensorGenerator(
        qdata,
        shape=shape,
        scale=scale, 
        zeropoint=0,
        is_model_input=is_model_input
    )


def generate_signed_symmetric_per_channel_quantized_tensor(
    config:TfliteLayerGeneratorConfig,
    name:str,
    input_tensor:TfliteTensorGenerator,
    qdim:int,
    qdrange:Tuple[int,int]=None,
    sparsity:float=None
) -> TfliteTensorGenerator:
    width, height, count = config.get_shape_config_entry(name, 3, throw_exception=True)
    data = config.get_config_entry(f'{name}_data')
    drange = config.get_range_config_entry(f'{name}_range') or config.get_range_config_entry('range')
    qdrange = qdrange or (kAsymmetricInt8Min, kAsymmetricInt8Max)

    input_depth = input_tensor.shape[-1]

    if qdim == 0:
        # shape = filter-count=output-depth, filter-height, filter-width, input-depth
        tensor_shape = (count, height, width, input_depth)
    elif qdim == 3:
        # shape = <placeholder>', 'height', 'width', 'input-depth * depth_multiplier'
        tensor_shape = (1, height, width, input_depth * count)
    else:
        raise Exception('Quantization dim not supported')

    data = _generate_tensor_data(shape=tensor_shape, data=data, drange=drange)
    data_flat = data.reshape(-1)

    input_size = len(data_flat)
    channel_count = tensor_shape[qdim]
    per_channel_size = input_size // channel_count
    scaling_factors = [0.] * channel_count
    zeropoints = [0] * channel_count

    if qdim == 0:
        stride = 1 
        channel_stride = per_channel_size
    elif qdim == 3:
        stride = channel_count
        channel_stride = 1
    else:
        raise Exception('qdim must be 0 or 3')

    # Calculate scales for each channel.
    for channel in range(channel_count):
        min_value = 0
        max_value = 0

        for i in range(per_channel_size):   
            idx = channel * channel_stride + i * stride
            min_value = min(min_value, data_flat[idx])
            max_value = max(max_value, data_flat[idx])

        scaling_factors[channel] = max(abs(min_value), abs(max_value)) / qdrange[1]

    qdata = np.empty(tensor_shape, dtype=np.int8)
    qdata_flat = qdata.reshape(-1)

    for channel in range(channel_count):
        for i in range(per_channel_size):
            idx = channel * channel_stride + i * stride
            quantized_value = round(data_flat[idx] / scaling_factors[channel])
            # Clamp: just in case some odd numeric offset.
            qdata_flat[idx] = min(qdrange[1], max(qdrange[0], quantized_value))

    return TfliteTensorGenerator(
        qdata,
        scale=scaling_factors, 
        zeropoint=zeropoints,
        qdim=qdim
    )


def generate_bias_tensor_config(
    config:TfliteLayerGeneratorConfig, 
    input_tensor:TfliteTensorGenerator,
    filters_tensor:TfliteTensorGenerator,
    qdim:int=0
) -> TfliteTensorGenerator:
    if 'add_bias' not in config or not config['add_bias']:
        return None 

    channel_count = filters_tensor.shape[qdim]
    bias_shape = (channel_count,)
    data =  config.get_config_entry(f'bias_data')
    drange = config.get_range_config_entry(f'bias_range') or config.get_range_config_entry('range')

    data = _generate_tensor_data(bias_shape, data, drange)

    input_scale = input_tensor.quantization.scale[0]
    weight_scales = filters_tensor.quantization.scale
    scaling_factors = [0.] * channel_count
    zeropoints = [0] * channel_count

    for i in range(channel_count):
        scaling_factors[i] = input_scale * weight_scales[i]

    qdata = np.empty(data.shape, dtype=np.int32)
    qdata_flat = qdata.reshape(-1)
    data_flat = data.reshape(-1)
    
    for i in range(channel_count):
        qdata_flat[i] = _symmetric_quantize_float_to_int32(data_flat[i], scaling_factors[i])

    return TfliteTensorGenerator( 
        qdata,
        zeropoint=zeropoints,
        scale=scaling_factors,
        qdim=qdim
    )


def generate_output_tensor(
    config:TfliteLayerGeneratorConfig, 
    input_tensor:TfliteTensorGenerator,
    filters_tensor:TfliteTensorGenerator,
    stride_width:int, 
    stride_height:int,
    padding:str
) -> TfliteTensorGenerator:
    input_shape = input_tensor.shape 
    filters_shape = filters_tensor.shape 

    batches = input_shape[0]
    in_h = input_shape[1]
    in_w = input_shape[2]
    filter_h = filters_shape[1]
    filter_w = filters_shape[2]
    channel_count = filters_tensor.shape[filters_tensor.quantization.quantization_dimension]
    out_w, out_h = compute_output_size(stride_width, stride_height, in_w, in_h, filter_w, filter_h, padding)
    output_shape = (batches, out_h, out_w, channel_count)
    return generate_asymmetric_quantized_tensor(
        config, 
        'output', 
        shape=output_shape,
        generate_random=False,
        default_scale=1.0,
        default_zeropoint=0,
    )


def _generate_tensor_data(
    shape, 
    data, 
    drange=None,
    generate_random=True
):
    if data is None and generate_random:
        drange = drange or [-1, 1]
        np.random.seed(0)
        return np.random.uniform(low=drange[0], high=drange[1], size=shape)
    
    if data is None:
        return None 

    data = np.asarray(data)
    data = data.reshape(shape)
    return data.astype(np.float32)


int8_range = np.iinfo(np.int8)
int32_range = np.iinfo(np.int32)

def _asymmetric_quantize_float_to_int8(fvalue, scale, zero_point):
    result = int(round(fvalue / scale))  + zero_point
    return max(int8_range.min, min(int8_range.max, result))


def _symmetric_quantize_float_to_int32(fvalue, scale):
    result = int(round(fvalue / scale))
    return max(int32_range.min + 1, min(int32_range.max, result))


def _compute_out_dim(padding, i, f, s):
    if padding.lower() == 'same':
        return (i + s - 1) // s 

    elif padding.lower() == 'valid':
        return (i + s - f) // s

    else:
        return 0
    
def compute_output_size(stride_w, stride_h, in_w, in_h, filter_w, filter_h, padding):
    out_w = _compute_out_dim(padding, in_w, filter_w, stride_w)
    out_h = _compute_out_dim(padding, in_h, filter_h, stride_h)

    return (out_w, out_h)


