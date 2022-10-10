
from typing import Union
from mltk.core.tflite_model.tflite_schema import BuiltinOperator
from mltk.core import TfliteConv2dLayer
from .layer import Layer, KerasLayer, TfliteLayer


class Conv2D(Layer):
    
    def __init__(self):
        Layer.__init__(self, ('Conv2D', BuiltinOperator.CONV_2D))
        
        
    def process(self, layer:Union[TfliteLayer,KerasLayer]):
        if isinstance(layer, TfliteConv2dLayer):
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
        self.name = layer.name
        self.macs = w_h * w_w * in_depth * out_w * out_h * out_depth
        self.ops = self.macs * 2 
        
        if  layer.use_bias:
            self.ops += out_w * out_h * out_depth


class Conv2DTranspose(Layer):
    
    def __init__(self):
        Layer.__init__(self, 'Conv2DTranspose')
        
        
    def process(self, layer):
        if layer.data_format == "channels_first":
            _, in_depth, _, _ = layer.input_shape
            _, out_depth, out_h, out_w, = layer.output_shape
        elif layer.data_format == "channels_last":
            _, _, _, in_depth = layer.input_shape
            _, out_h, out_w, out_depth = layer.output_shape
        
        w_h, w_w =  layer.kernel_size

        self.name = layer.name
        self.macs = w_h * w_w * in_depth * out_w * out_h * out_depth
        self.ops = self.macs * 2 
        
        if layer.use_bias:
            self.ops += out_w * out_h * out_depth



class Conv2DTransposeShared(Layer):
    
    def __init__(self):
        Layer.__init__(self, 'Conv2DTransposeShared')
        
        
    def process(self, layer):
        _, _, _, in_depth = layer.layer.input_shape
        _, out_h, out_w, out_depth = layer.layer.output_shape
        
        w_h, w_w =  layer.layer.kernel_size

        self.name = layer.name
        self.macs = w_h * w_w * in_depth * out_w * out_h * out_depth
        self.ops = self.macs * 2 
        
        if layer.layer.use_bias:
            self.ops += out_w * out_h * out_depth





Conv2D()
Conv2DTranspose()
Conv2DTransposeShared()