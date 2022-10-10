
from typing import Union
from mltk.core.tflite_model.tflite_schema import BuiltinOperator
from .layer import Layer, flat_size, KerasLayer, TfliteLayer


class Pad(Layer):
    
    def __init__(self):
        Layer.__init__(self, ('Pad', BuiltinOperator.PAD, BuiltinOperator.PADV2))
    
    def process(self, layer:Union[TfliteLayer,KerasLayer]):
        self.name = layer.name
        self.macs = 0
        # https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/kernels/internal/reference/pad.h
        # 1 mv
        # 10/2 =  5 comparisions
        if isinstance(layer, KerasLayer):
            if isinstance(layer.output_shape[0], int):
                shape = layer.output_shape
            else:
                shape = layer.output_shape[0]
        else:
            shape = layer.outputs[0].shape

        self.ops = flat_size(shape[1:]) * 6


Pad()


