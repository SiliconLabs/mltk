from typing import List, Tuple, Union, Dict
import math
import copy
import numpy as np
import tensorflow_lite_support.metadata.schema_py_generated as tflite_fb

from mltk.utils.python import (find_object_key_with_value, find_object_value_with_key, as_list)


from ... import (
    TfliteOpCode,
    TfliteTensor,
    TfliteShape,
    TfliteLayer, 
    TfliteLayerOptions,
    TfliteQuantization,
    TfliteAddLayer, 
    TfliteConv2dLayer, 
    TfliteTransposeConvLayer,
    TfliteDepthwiseConv2dLayer, 
    TfliteFullyConnectedLayer, 
    TflitePooling2dLayer,
    TfliteMulLayer,
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
        self._index = -1 
        self._shape = TfliteShape(shape or data.shape)
        self._name = ''
        self._dtype = np.int8 if data is None else data.dtype 
        self._data = data 
        self._quantization = TfliteQuantization(None)
        self._quantization._scale = as_list(scale)
        self._quantization._zeropoint = as_list(zeropoint)
        self._quantization._quantization_dimension = qdim
        self.is_model_input = is_model_input

    @property
    def dtype_str(self) -> str:
        """Tensor data type as a string"""
        if hasattr(self._dtype, 'name'):
            return self._dtype.name 
        return self._dtype.__name__.replace('numpy.', '')


    @property
    def tflite_type(self) -> int:
        if self._dtype == np.uint8:
            return tflite_fb.TensorType.UINT8
        if self._dtype == np.int8:
            return tflite_fb.TensorType.INT8
        if self._dtype == np.int16:
            return tflite_fb.TensorType.INT16
        if self._dtype == np.int32:
            return tflite_fb.TensorType.INT32
        if self._dtype == np.int64:
            return tflite_fb.TensorType.INT64
        if self._dtype == np.float32:
            return tflite_fb.TensorType.FLOAT32

        raise NotImplementedError('Data type not supported')


class TfliteLayerGenerator(TfliteLayer):
    def __init__(self, opcode:TfliteOpCode):
        # NOTE: We purposely don't call TfliteLayer.__init__() 
        self._index = 0
        self._opcode:TfliteOpCode = opcode 
        self._opcode_version = 0
        self._opcode_str:str = find_object_key_with_value(TfliteOpCode, self.opcode)
        self._inputs:List[TfliteTensorGenerator] = []
        self._outputs:List[TfliteTensorGenerator] = []
        self._options:TfliteLayerOptions = None


    @property
    def inputs(self) -> List[TfliteTensorGenerator]:
        return self._inputs

    @property
    def outputs(self) -> List[TfliteTensorGenerator]:
        return self._outputs

    def generate_options(self) -> Tuple:
        raise NotImplementedError 


class TfliteConv2dLayerGenerator(TfliteLayerGenerator, TfliteConv2dLayer):

    def __init__(self, config:TfliteLayerGeneratorConfig):
        """
        Configuration values:
        input : input tensor shape, <batches> x <height> x <width> x <depth>
        filters : filters shape, <width> x <height> x <filter count>
        padding : input padding: same or valid
        stride: <width> x <height>
        activation: output activation : none, relu, or relu6
        add_bias: add a bias tensor
        range: range of generated input/filter,bias tensor data: <min>,<max>

        """
        super().__init__(TfliteOpCode.CONV_2D)

        self._options = TfliteLayerOptions()
        self._options.stride_width, self._options.stride_height = config.stride
        self._options.padding = config.padding
        self._options.activation = config.activation

        input_tensor = generate_asymmetric_quantized_tensor(
            config, 
            'input',
            default_scale=0.5,
            default_zeropoint=0,
            is_model_input=True
        )
        filters_tensor = generate_signed_symmetric_per_channel_quantized_tensor(
            config,
            'filters',
            input_tensor,
            qdim=0,
            qdrange=(-127,127)
        )
        bias_tensor = generate_bias_tensor_config(
            config, 
            input_tensor, 
            filters_tensor
        )
        output_tensor = generate_output_tensor(
            config, 
            input_tensor, 
            filters_tensor,
            stride_width=self._options.stride_width, 
            stride_height=self._options.stride_height,
            padding=self._options.padding
        )

        filters_shape = filters_tensor.shape
        self._kernel_size = (filters_shape[1], filters_shape[2])

        self._inputs.append(input_tensor)
        self._inputs.append(filters_tensor)
        if bias_tensor is not None:
            self._inputs.append(bias_tensor)
            self._bias_data = bias_tensor.data
        else:
            self._bias_data = None
        self._outputs.append(output_tensor)
        

    def generate_options(self) -> Tuple:
        options_type = tflite_fb.BuiltinOptions.Conv2DOptions
        options = tflite_fb.Conv2DOptionsT()
        options.padding = find_object_value_with_key(
            tflite_fb.Padding, 
            self._options.padding,
            ignore_case=True
        )
        options.fusedActivationFunction = find_object_value_with_key(
            tflite_fb.ActivationFunctionType, 
            self._options.activation,
            ignore_case=True
        )
        options.strideW = self._options.stride_width
        options.strideH = self._options.stride_height
        return options_type, options



class TfliteTransposeConvLayerGenerator(TfliteLayerGenerator, TfliteTransposeConvLayer):

    def __init__(self, config:TfliteLayerGeneratorConfig):
        """
        Configuration values:
        input : input tensor shape, <batches> x <height> x <width> x <depth>
        filters : filters shape, <width> x <height> x <filter count>
        padding : input padding: same or valid
        stride: <width> x <height>
        add_bias: add a bias tensor
        range: range of generated input/filter,bias tensor data: <min>,<max>

        """
        super().__init__(TfliteOpCode.TRANSPOSE_CONV)

        self._options = TfliteLayerOptions()
        self._options.stride_width, self._options.stride_height = config.stride
        self._options.padding = config.padding

        input_tensor = generate_asymmetric_quantized_tensor(
            config, 
            'input',
            default_scale=0.5,
            default_zeropoint=0,
            is_model_input=True
        )
        filters_tensor = generate_signed_symmetric_per_channel_quantized_tensor(
            config,
            'filters',
            input_tensor,
            qdim=0,
            qdrange=(-127,127)
        )
        bias_tensor = generate_bias_tensor_config(
            config, 
            input_tensor, 
            filters_tensor
        )
        output_tensor = generate_output_tensor(
            config, 
            input_tensor, 
            filters_tensor,
            stride_width=self._options.stride_width, 
            stride_height=self._options.stride_height,
            padding=self._options.padding
        )

        filters_shape = filters_tensor.shape
        self._kernel_size = (filters_shape[1], filters_shape[2])

        out_shape_data = np.ndarray(shape=(len(output_tensor.shape,)), dtype=np.int32)
        for i, d in enumerate(output_tensor.shape):
            out_shape_data[i] = output_tensor.shape[1]

        output_shape_tensor = TfliteTensorGenerator( 
            out_shape_data,
            is_model_input=True
        )

        self._inputs.append(output_shape_tensor)
        self._inputs.append(filters_tensor)
        self._inputs.append(input_tensor)
        if bias_tensor is not None:
            self._inputs.append(bias_tensor)
            self._bias_data = bias_tensor.data
        else:
            self._bias_data = None
        self._outputs.append(output_tensor)
        

    def generate_options(self) -> Tuple:
        options_type = tflite_fb.BuiltinOptions.TransposeConvOptions
        options = tflite_fb.TransposeConvOptionsT()
        options.padding = find_object_value_with_key(
            tflite_fb.Padding, 
            self._options.padding,
            ignore_case=True
        )
        options.strideW = self._options.stride_width
        options.strideH = self._options.stride_height
        return options_type, options


class TfliteAddLayerGenerator(TfliteLayerGenerator, TfliteAddLayer):
    def __init__(self, config:TfliteLayerGeneratorConfig):
        """
        Configuration values:
        input : input1, input2, and output tensor shape, <batches> x <height> x <width> x <depth>
        activation: output activation : none, relu, or relu6
        range: range of generated input/filter,bias tensor data: <min>,<max>
        """
        super().__init__(TfliteOpCode.ADD)
        self._options = TfliteLayerOptions()
        self._options.activation = config.activation

        input1_tensor = generate_asymmetric_quantized_tensor(
            config, 
            'input', 
            default_scale=0.5,
            default_zeropoint=0,
            is_model_input=True,
        )
        input2_tensor = generate_asymmetric_quantized_tensor(
            config, 
            'input2',
            shape=input1_tensor.shape,
            default_scale=0.5,
            default_zeropoint=0,
            is_model_input=True,
        )
        output_tensor = generate_asymmetric_quantized_tensor(
            config, 
            'output',
            shape=input2_tensor.shape,
            generate_random=False, 
            default_scale=0.5,
            default_zeropoint=0,
        )

        self._inputs.append(input1_tensor)
        self._inputs.append(input2_tensor)
        self._outputs.append(output_tensor)
        

    def generate_options(self) -> Tuple:
        options_type = tflite_fb.BuiltinOptions.AddOptions
        options = tflite_fb.AddOptionsT()
        options.fusedActivationFunction = find_object_value_with_key(
            tflite_fb.ActivationFunctionType, 
            self._options.activation,
            ignore_case=True
        )
        return options_type, options


class TfliteFullyConnectedLayerGenerator(TfliteLayerGenerator, TfliteFullyConnectedLayer):

    def __init__(self, config:TfliteLayerGeneratorConfig):
        """
        Configuration values:
        batches : # number of batches
        features : number of "features" in input tensor
        classes : number of "classes" in output tensor
        activation: output activation : none, relu, or relu6
        add_bias: add a bias tensor
        range: range of generated input/filter,bias tensor data: <min>,<max>
        """
        super().__init__(TfliteOpCode.FULLY_CONNECTED)

        self._options = TfliteLayerOptions()
        self._options.activation = config.activation

        n_batches = config.get_config_entry('batches') or 1
        self._n_classes = n_classes = config.get_config_entry('classes', throw_exception=True)
        n_features = config.get_config_entry('features', throw_exception=True)
        use_bias = config.get_config_entry('add_bias', throw_exception=False)

        input_tensor = generate_asymmetric_quantized_tensor(
            config, 
            'input', 
            shape=(n_batches, n_features),
            default_scale=0.5,
            default_zeropoint=0,
            is_model_input=True,
        )
        weights_tensor = generate_signed_symmetric_quantized_tensor(
            config, 
            'weights',
            shape=(n_classes, n_features),
            dtype=np.int8,
            qdrange=(-127,127)
        )
        if use_bias:
            bias_tensor = generate_signed_symmetric_quantized_tensor(
                config, 
                'bias',
                shape=(1, n_classes),
                generate_random=True, 
                scale=input_tensor.quantization.scale[0] * weights_tensor.quantization.scale[0],
                dtype=np.int32
            )
        else:
            bias_tensor = None 
        
        output_tensor = generate_asymmetric_quantized_tensor(
            config, 
            'output',
            shape=(n_batches, n_classes),
            default_scale=0.5,
            default_zeropoint=0,
            generate_random=False
        )
        self._inputs.append(input_tensor)
        self._inputs.append(weights_tensor)
        if bias_tensor is not None:
            self._inputs.append(bias_tensor)
            self._bias_data = bias_tensor.data 
        else:
            self._bias_data = None
        self._outputs.append(output_tensor)
        

    @property
    def units(self) -> int:
        """Number of neurons"""
        return self._n_classes


    def generate_options(self) -> Tuple:
        options_type = tflite_fb.BuiltinOptions.FullyConnectedOptions
        options = tflite_fb.FullyConnectedOptionsT()
        options.fusedActivationFunction = find_object_value_with_key(
            tflite_fb.ActivationFunctionType, 
            self._options.activation,
            ignore_case=True
        )
        return options_type, options


class TfliteDepthwiseConv2dLayerGenerator(TfliteLayerGenerator, TfliteDepthwiseConv2dLayer):

    def __init__(self, config:TfliteLayerGeneratorConfig):
        """
        Configuration values:
        input : input tensor shape, <batches> x <height> x <width> x <depth>
        filters : filters shape, <width> x <height> x <depth multiplier>
        padding : input padding: same or valid
        stride: <width> x <height>
        activation: output activation : none, relu, or relu6
        add_bias: add a bias tensor
        range: range of generated input/filter,bias tensor data: <min>,<max>
        """
        super().__init__(TfliteOpCode.DEPTHWISE_CONV_2D)

        self._options = TfliteLayerOptions()
        self._options.stride_width, self._options.stride_height = config.stride
        self._options.padding = config.padding
        self._options.activation = config.activation

        input_tensor = generate_asymmetric_quantized_tensor(
            config, 
            'input',
            default_scale=0.5,
            default_zeropoint=0,
            is_model_input=True
        )
        filters_tensor = generate_signed_symmetric_per_channel_quantized_tensor(
            config,
            'filters',
            input_tensor,
            qdim=3,
            qdrange=(-127,127)
        )
        bias_tensor = generate_bias_tensor_config(
            config, 
            input_tensor, 
            filters_tensor,
            qdim=3,
        )
        output_tensor = generate_output_tensor(
            config, 
            input_tensor, 
            filters_tensor,
            stride_width=self._options.stride_width, 
            stride_height=self._options.stride_height,
            padding=self._options.padding
        )

        filters_shape = filters_tensor.shape
        self._kernel_size = (filters_shape[1], filters_shape[2])
        self._options.multiplier = filters_shape[3] // input_tensor.shape[-1]

        self._inputs.append(input_tensor)
        self._inputs.append(filters_tensor)
        if bias_tensor is not None:
            self._inputs.append(bias_tensor)
            self._bias_data = bias_tensor.data
        else:
            self._bias_data = None
        self._outputs.append(output_tensor)
        

    def generate_options(self) -> Tuple:
        options_type = tflite_fb.BuiltinOptions.DepthwiseConv2DOptions
        options = tflite_fb.DepthwiseConv2DOptionsT()
        options.padding = find_object_value_with_key(
            tflite_fb.Padding, 
            self._options.padding,
            ignore_case=True
        )
        options.depthMultiplier = self._options.multiplier
        options.fusedActivationFunction = find_object_value_with_key(
            tflite_fb.ActivationFunctionType, 
            self._options.activation,
            ignore_case=True
        )
        options.strideW = self._options.stride_width
        options.strideH = self._options.stride_height
        return options_type, options


class TflitePooling2dLayerGenerator(TfliteLayerGenerator, TflitePooling2dLayer):
    def __init__(self, opcode:TfliteOpCode, config:TfliteLayerGeneratorConfig):
        super().__init__(opcode)

        self._options = TfliteLayerOptions()
        self._options.activation = config.activation
        self._options.stride_width, self._options.stride_height = config.stride
        self._options.filter_width, self._options.filter_height = config.filter
        self._options.padding = config.padding

        input_tensor = generate_asymmetric_quantized_tensor(
            config, 
            'input', 
            default_scale=0.5,
            default_zeropoint=0,
            is_model_input=True,
        )

        batches, in_h, in_w, in_d = input_tensor.shape

        out_w, out_h = _compute_output_size(
            self._options.stride_width, self._options.stride_height, 
            in_w, in_h, 
            self._options.filter_width, self._options.filter_height, 
            self._options.padding
        )
        output_shape = (batches, out_h, out_w, in_d)
        output_tensor = generate_asymmetric_quantized_tensor(
            config, 
            'output',
            shape=output_shape,
            generate_random=False, 
            default_scale=0.5,
            default_zeropoint=0,
        )

        self._inputs.append(input_tensor)
        self._outputs.append(output_tensor)
        

    def generate_options(self) -> Tuple:
        options_type = tflite_fb.BuiltinOptions.Pool2DOptions
        options = tflite_fb.Pool2DOptionsT()
        options.padding = find_object_value_with_key(
            tflite_fb.Padding, 
            self._options.padding,
            ignore_case=True
        )
        options.fusedActivationFunction = find_object_value_with_key(
            tflite_fb.ActivationFunctionType, 
            self._options.activation,
            ignore_case=True
        )
        options.strideW = self._options.stride_width
        options.strideH = self._options.stride_height
        options.filterWidth = self._options.filter_width
        options.filterHeight = self._options.filter_height
        return options_type, options


class TfliteAveragePool2dLayerGenerator(TflitePooling2dLayerGenerator):
    def __init__(self, config:TfliteLayerGeneratorConfig):
        """
        Configuration values:
        input : input tensor shape, <batches> x <height> x <width> x <depth>
        filter : filters shape, <width> x <height>
        padding : input padding: same or valid
        stride: <width> x <height>
        activation: output activation : none, relu, or relu6
        range: range of generated input/filter,bias tensor data: <min>,<max>
        """
        super().__init__(TfliteOpCode.AVERAGE_POOL_2D, config)

        
class TfliteMaxPool2dLayerGenerator(TflitePooling2dLayerGenerator):
    def __init__(self, config:TfliteLayerGeneratorConfig):
        """
        Configuration values:
        input : input tensor shape, <batches> x <height> x <width> x <depth>
        filter : filters shape, <width> x <height>
        padding : input padding: same or valid
        stride: <width> x <height>
        activation: output activation : none, relu, or relu6
        range: range of generated input/filter,bias tensor data: <min>,<max>
        """
        super().__init__(TfliteOpCode.MAX_POOL_2D, config)


class TfliteMulLayerGenerator(TfliteLayerGenerator, TfliteMulLayer):
    def __init__(self, config:TfliteLayerGeneratorConfig):
        """
        Configuration values:
        input : input1, input2, and output tensor shape, <batches> x <height> x <width> x <depth>
        activation: output activation : none, relu, or relu6
        range: range of generated input/filter,bias tensor data: <min>,<max>
        """
        super().__init__(TfliteOpCode.MUL)
        self._options = TfliteLayerOptions()
        self._options.activation = config.activation

        input1_tensor = generate_asymmetric_quantized_tensor(
            config, 
            'input', 
            default_scale=0.5,
            default_zeropoint=0,
            is_model_input=True,
        )
        input2_tensor = generate_asymmetric_quantized_tensor(
            config, 
            'input2',
            shape=input1_tensor.shape,
            default_scale=0.5,
            default_zeropoint=0,
            is_model_input=True,
        )
        output_tensor = generate_asymmetric_quantized_tensor(
            config, 
            'output',
            shape=input2_tensor.shape,
            generate_random=False, 
            default_scale=0.5,
            default_zeropoint=0,
        )

        self._inputs.append(input1_tensor)
        self._inputs.append(input2_tensor)
        self._outputs.append(output_tensor)
        

    def generate_options(self) -> Tuple:
        options_type = tflite_fb.BuiltinOptions.MulOptions
        options = tflite_fb.MulOptionsT()
        options.fusedActivationFunction = find_object_value_with_key(
            tflite_fb.ActivationFunctionType, 
            self._options.activation,
            ignore_case=True
        )
        return options_type, options


def opcode_to_layer_generator_class(opcode:TfliteOpCode) -> TfliteLayerGenerator:
    if opcode in _LAYER_CLASSES:
        return _LAYER_CLASSES[opcode]

    return None

_LAYER_CLASSES:Dict[TfliteOpCode, TfliteLayerGenerator] = {}
_LAYER_CLASSES[TfliteOpCode.ADD] = TfliteAddLayerGenerator
_LAYER_CLASSES[TfliteOpCode.CONV_2D] = TfliteConv2dLayerGenerator
_LAYER_CLASSES[TfliteOpCode.TRANSPOSE_CONV] = TfliteTransposeConvLayerGenerator
_LAYER_CLASSES[TfliteOpCode.FULLY_CONNECTED] = TfliteFullyConnectedLayerGenerator
_LAYER_CLASSES[TfliteOpCode.DEPTHWISE_CONV_2D] = TfliteDepthwiseConv2dLayerGenerator
_LAYER_CLASSES[TfliteOpCode.AVERAGE_POOL_2D] = TfliteAveragePool2dLayerGenerator
_LAYER_CLASSES[TfliteOpCode.MAX_POOL_2D] = TfliteMaxPool2dLayerGenerator
_LAYER_CLASSES[TfliteOpCode.MUL] = TfliteMulLayerGenerator




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
    qdrange:Tuple[int,int]=None
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
    out_w, out_h = _compute_output_size(stride_width, stride_height, in_w, in_h, filter_w, filter_h, padding)
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
    
def _compute_output_size(stride_w, stride_h, in_w, in_h, filter_w, filter_h, padding):
    out_w = _compute_out_dim(padding, in_w, filter_w, stride_w)
    out_h = _compute_out_dim(padding, in_h, filter_h, stride_h)

    return (out_w, out_h)


