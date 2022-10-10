from typing import Union
from mltk.core import TfliteDepthwiseConv2dLayer
from mltk.core.tflite_model.tflite_schema import BuiltinOperator
from .layer import Layer, KerasLayer, TfliteLayer


class DepthwiseConv2D(Layer):
    
    def __init__(self):
        Layer.__init__(self, ('DepthwiseConv2D', BuiltinOperator.DEPTHWISE_CONV_2D))
        
        
    def process(self, layer:Union[TfliteLayer,KerasLayer]):
        if isinstance(layer, TfliteDepthwiseConv2dLayer):
            _, _, _, in_depth = layer.input_data.shape
            _, out_h, out_w, out_depth = layer.output_data.shape

        else:
            if layer.data_format == "channels_first":
                _, in_depth, _, _ = layer.input_shape
                _, out_depth, out_h, out_w, = layer.output_shape
            elif layer.data_format == "channels_last":
                _, _, _, in_depth = layer.input_shape
                _, out_h, out_w, out_depth = layer.output_shape

        w_h, w_w =  layer.kernel_size
        depth_multiplier = out_depth // in_depth
       
        self.name = layer.name
        self.macs = w_h * w_w * depth_multiplier * in_depth * out_w * out_h
        self.ops = self.macs * 2 
        
        if layer.use_bias:
            self.ops += depth_multiplier * in_depth * out_w * out_h


DepthwiseConv2D()
