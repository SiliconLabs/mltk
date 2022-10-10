# pylint: disable=import-outside-toplevel,unused-import
import copy
from typing import Union


from tensorflow.keras.layers import Layer as KerasLayer
from mltk.core import TfliteLayer

from .layer import Layer, SUPPORTED_LAYERS


def parse_layer(model_layer: Union[TfliteLayer, KerasLayer]) -> Layer:
    if isinstance(model_layer, KerasLayer):
        opcode = model_layer.__class__.__name__
    else:
        opcode = model_layer.opcode

    for l in SUPPORTED_LAYERS:
        if opcode in l.opcodes:
            layer = copy.deepcopy(l)
            layer.process(model_layer)
            return layer
    
    return None


def load_layers():
    if len(SUPPORTED_LAYERS) > 0: 
        return 
    
    from . import conv
    from . import depthwise_conv
    from . import activation
    from . import dense 
    from . import pooling
    from . import add
    from . import quantize
    from . import pad
    from . import reshape
    from . import ignored
    from . import multiply
