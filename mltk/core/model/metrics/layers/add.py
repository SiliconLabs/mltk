from typing import Union
from mltk.core.tflite_model.tflite_schema import BuiltinOperator
from .layer import Layer, flat_size, KerasLayer, TfliteLayer


class Add(Layer):
    
    def __init__(self):
        Layer.__init__(self, ('Add', BuiltinOperator.ADD))
    
    def process(self, layer:Union[TfliteLayer,KerasLayer]):
        self.name = layer.name
        self.macs = 0
        if isinstance(layer, KerasLayer):
            input_shape = layer.input_shape[0]
        else:
            input_shape = layer.inputs[0].shape
        self.ops = flat_size(input_shape[1:])

Add()

