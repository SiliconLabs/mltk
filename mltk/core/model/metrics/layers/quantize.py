from typing import Union
from mltk.core.tflite_model.tflite_schema import BuiltinOperator
from .layer import Layer, flat_size, KerasLayer, TfliteLayer


class Quantize(Layer):
    
    def __init__(self):
        Layer.__init__(self, ('Quantize', BuiltinOperator.QUANTIZE))
    
    def process(self, layer:Union[TfliteLayer,KerasLayer]):
        self.name = layer.name
        self.macs = 0
        # https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/kernels/internal/reference/quantize.h#L31
        # 1 divide
        # 1 add
        # 1 min
        # 1 max
        if isinstance(layer, KerasLayer):
            input_shape = layer.input_shape
        else:
            input_shape = layer.inputs[0].shape

        self.ops = flat_size(input_shape[1:]) * 4


class Dequantize(Layer):
    
    def __init__(self):
        Layer.__init__(self, ('Dequantize', BuiltinOperator.DEQUANTIZE))
    
    def process(self, layer:Union[TfliteLayer,KerasLayer]):
        self.name = layer.name
        self.macs = 0
        # https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/kernels/internal/reference/dequantize.h
        # 1 multiply
        # 1 substract

        if isinstance(layer, KerasLayer):
            input_shape = layer.input_shape
        else:
            input_shape = layer.inputs[0].shape

        self.ops = flat_size(input_shape[1:]) * 2


Quantize()
Dequantize()

