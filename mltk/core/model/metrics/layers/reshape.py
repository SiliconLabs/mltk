from typing import Union
from mltk.core.tflite_model.tflite_schema import BuiltinOperator
from mltk.core import TfliteReshapeLayer
from .layer import Layer, KerasLayer, TfliteLayer


class Reshape(Layer):
    
    def __init__(self):
        Layer.__init__(self, ('Reshape', BuiltinOperator.RESHAPE))
    
    def process(self, layer:Union[TfliteLayer,KerasLayer]):
        self.name = layer.name
        self.macs = 0
        self.ops = 0

        if isinstance(layer, TfliteReshapeLayer):
            # If a copy is required
            # Then 1 op for the mv operation
            if layer.requires_copy:
                self.ops = layer.inputs[0].shape.flat_size
 
Reshape()


