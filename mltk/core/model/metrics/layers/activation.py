from typing import Union
from mltk.core.tflite_model.tflite_schema import BuiltinOperator
from .layer import Layer, flat_size, KerasLayer, TfliteLayer


class Activation(Layer):
    
    def __init__(self):
        Layer.__init__(self, 'Activation')
        
    def process(self, layer:Union[TfliteLayer,KerasLayer]):
        self.name = layer.name
        if isinstance(layer, KerasLayer):
            output_shape = layer.output_shape
        else:
            output_shape = layer.outputs[0].shape

        self.ops = flat_size(output_shape[1:]) * 2 # min(max(lower, x), upper)


class ReLU(Layer):
    
    def __init__(self):
        Layer.__init__(self, ('ReLU', BuiltinOperator.RELU))
        
    def process(self, layer:Union[TfliteLayer,KerasLayer]):
        self.name = layer.name
        if isinstance(layer, KerasLayer):
            output_shape = layer.output_shape
        else:
            output_shape = layer.outputs[0].shape

        self.ops = flat_size(output_shape[1:]) * 2 # min(max(lower, x), upper)


class Softmax(Layer):
    
    def __init__(self):
        Layer.__init__(self, ('Softmax', BuiltinOperator.SOFTMAX))
        
    def process(self, layer:Union[TfliteLayer,KerasLayer]):
        self.name = layer.name

        if isinstance(layer, KerasLayer):
            input_shape = layer.input_shape
        else:
            input_shape = layer.inputs[0].shape
        
        nfeatures = flat_size(input_shape[1:])
        total_exp = nfeatures 
        total_add = nfeatures - 1
        total_div = nfeatures
        
        self.ops = total_div + total_exp


Activation()
ReLU()
Softmax()