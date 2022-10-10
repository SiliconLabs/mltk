from typing import Union
from mltk.core.tflite_model.tflite_schema import BuiltinOperator
from .layer import Layer, flat_size, KerasLayer, TfliteLayer


class AveragePooling2D(Layer):
    
    def __init__(self):
        Layer.__init__(self, ('AveragePooling2D', BuiltinOperator.AVERAGE_POOL_2D))
        
    def process(self, layer:Union[TfliteLayer,KerasLayer]):
        f_h, f_w  = layer.pool_size
        self.name = layer.name
        if isinstance(layer, KerasLayer):
            output_shape = layer.output_shape
        else:
            output_shape = layer.outputs[0].shape

        self.ops = flat_size(output_shape[1:]) * (f_h * f_w + 1) # + 1 for the division




class MaxPooling2D(Layer):
    
    def __init__(self):
        Layer.__init__(self, ('MaxPooling2D', BuiltinOperator.MAX_POOL_2D))
        
    def process(self, layer:Union[TfliteLayer,KerasLayer]):
        f_h, f_w  = layer.pool_size
        self.name = layer.name
        if isinstance(layer, KerasLayer):
            output_shape = layer.output_shape
        else:
            output_shape = layer.outputs[0].shape
        
        self.ops = flat_size(output_shape[1:]) * f_h * f_w






AveragePooling2D()
MaxPooling2D()