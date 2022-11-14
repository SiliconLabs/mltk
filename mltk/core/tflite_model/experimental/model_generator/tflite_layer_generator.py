from typing import List, Tuple, Union, Dict
import math
import numpy as np

from mltk.utils.python import find_object_key_with_value

from .utils import (
    TfliteLayerGeneratorConfig,
    TfliteTensorGenerator,
    generate_asymmetric_quantized_tensor,
    generate_signed_symmetric_quantized_tensor,
    generate_signed_symmetric_per_channel_quantized_tensor,
    generate_bias_tensor_config,
    generate_output_tensor,
    compute_output_size
)

from ... import (
    TfliteOpCode,
    TfliteTensor,
    TfliteQuantization,
    TfliteLayer, 
    TfliteLayerOptions,
    TfliteAddLayer,
    TfliteAddLayerOptions,
    TfliteConv2dLayer,
    TfliteConv2DLayerOptions,
    TfliteTransposeConvLayer,
    TfliteTransposeConvLayerOptions,
    TfliteFullyConnectedLayer,
    TfliteFullyConnectedLayerOptions,
    TfliteDepthwiseConv2dLayer,
    TfliteDepthwiseConv2DLayerOptions,
    TflitePooling2dLayer,
    TflitePool2DLayerOptions,
    TfliteReshapeLayer,
    TfliteQuantizeLayer,
    TfliteDequantizeLayer,
    TfliteMulLayer,
    TfliteMulLayerOptions,
)






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
        TfliteLayerGenerator.__init__(self, TfliteOpCode.CONV_2D)

        self._options = TfliteConv2DLayerOptions()
        self._options.stride_width, self._options.stride_height = config.stride
        self._options.padding_str = config.padding
        self._options.activation_str = config.activation

        input_tensor = generate_asymmetric_quantized_tensor(
            config, 
            'input',
            default_scale=0.5,
            default_zeropoint=0,
            is_model_input=True,
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
            padding=self._options.padding_str
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
        TfliteLayerGenerator.__init__(self, TfliteOpCode.TRANSPOSE_CONV)

        self._options = TfliteTransposeConvLayerOptions()
        self._options.stride_width, self._options.stride_height = config.stride
        self._options.padding_str = config.padding

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
            padding=self._options.padding_str
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
    


class TfliteAddLayerGenerator(TfliteLayerGenerator, TfliteAddLayer):
    def __init__(self, config:TfliteLayerGeneratorConfig):
        """
        Configuration values:
        input : input1, input2, and output tensor shape, <batches> x <height> x <width> x <depth>
        activation: output activation : none, relu, or relu6
        range: range of generated input/filter,bias tensor data: <min>,<max>
        """
        TfliteLayerGenerator.__init__(self, TfliteOpCode.ADD)
        self._options = TfliteAddLayerOptions()
        self._options.activation_str = config.activation

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
        TfliteLayerGenerator.__init__(self, TfliteOpCode.FULLY_CONNECTED)

        self._options = TfliteFullyConnectedLayerOptions()
        self._options.activation_str = config.activation

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
        TfliteLayerGenerator.__init__(self, TfliteOpCode.DEPTHWISE_CONV_2D)

        self._options = TfliteDepthwiseConv2DLayerOptions()
        self._options.stride_width, self._options.stride_height = config.stride
        self._options.padding_str = config.padding
        self._options.activation_str = config.activation

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
            padding=self._options.padding_str
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
        


class TflitePooling2dLayerGenerator(TfliteLayerGenerator, TflitePooling2dLayer):
    def __init__(self, opcode:TfliteOpCode, config:TfliteLayerGeneratorConfig):
        TfliteLayerGenerator.__init__(self, opcode)

        self._options = TflitePool2DLayerOptions()
        self._options.activation_str = config.activation
        self._options.stride_width, self._options.stride_height = config.stride
        self._options.filter_width, self._options.filter_height = config.filter
        self._options.padding_str = config.padding

        input_tensor = generate_asymmetric_quantized_tensor(
            config, 
            'input', 
            default_scale=0.5,
            default_zeropoint=0,
            is_model_input=True,
        )

        batches, in_h, in_w, in_d = input_tensor.shape

        out_w, out_h = compute_output_size(
            self._options.stride_width, self._options.stride_height, 
            in_w, in_h, 
            self._options.filter_width, self._options.filter_height, 
            self._options.padding_str
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
        TflitePooling2dLayerGenerator.__init__(self, TfliteOpCode.AVERAGE_POOL_2D, config)

        
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
        TflitePooling2dLayerGenerator.__init__(self, TfliteOpCode.MAX_POOL_2D, config)


class TfliteMulLayerGenerator(TfliteLayerGenerator, TfliteMulLayer):
    def __init__(self, config:TfliteLayerGeneratorConfig):
        """
        Configuration values:
        input : input1, input2, and output tensor shape, <batches> x <height> x <width> x <depth>
        activation: output activation : none, relu, or relu6
        range: range of generated input/filter,bias tensor data: <min>,<max>
        """
        super().__init__(TfliteOpCode.MUL)
        self._options = TfliteMulLayerOptions()
        self._options.activation_str = config.activation

        # For the MUL broadcasting implementation the shapes of both input 
        # tensors must be specified
        input1_shape = config.get_shape_config_entry("input1", 4)
        input2_shape = config.get_shape_config_entry("input2", 4)
        output_shape = config.get_shape_config_entry("output", 4)

        # For support of the MUL layer without broadcasting
        if input1_shape is None or input2_shape is None or output_shape is None:
            input_shape = config.get_shape_config_entry("input", 4, True)
            input1_shape = input_shape
            input2_shape = input_shape
            output_shape = input_shape

        input1_tensor = generate_asymmetric_quantized_tensor(
            config, 
            'input1',
            shape=input1_shape,
            default_scale=0.5,
            default_zeropoint=0,
            is_model_input=True,
        )
        input2_tensor = generate_asymmetric_quantized_tensor(
            config, 
            'input2',
            shape=input2_shape,
            default_scale=0.5,
            default_zeropoint=0,
            is_model_input=True,
        )
        output_tensor = generate_asymmetric_quantized_tensor(
            config, 
            'output',
            shape=output_shape,
            generate_random=False, 
            default_scale=0.5,
            default_zeropoint=0,
        )

        self._inputs.append(input1_tensor)
        self._inputs.append(input2_tensor)
        self._outputs.append(output_tensor)
        




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


