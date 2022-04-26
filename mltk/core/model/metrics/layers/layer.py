from typing import Union
from abc import ABC, abstractmethod
from collections import namedtuple

from mltk.core import TfliteLayer
from tensorflow.keras.layers import Layer as KerasLayer


SUPPORTED_LAYERS = []


LayerMetrics = namedtuple('LayerMetrics', ['macs', 'ops'])


class Layer(ABC):
    def __init__(self, opcodes: tuple):
        global SUPPORTED_LAYERS # pylint: disable=global-statement
        SUPPORTED_LAYERS.append(self)
        if not isinstance(opcodes, (list,tuple)):
            opcodes = (opcodes,)
        self.opcodes = opcodes
        self.metrics = LayerMetrics(0, 0)
        self.name = None

    @property
    def macs(self) -> int:
        """# Multiply-accumulate operations"""
        return self.metrics.macs
    @macs.setter
    def macs(self, v: int):
        self.metrics = LayerMetrics(v, self.ops)
    
    @property
    def ops(self) -> int:
        """# operations"""
        return self.metrics.ops
    @ops.setter
    def ops(self, v: int):
        self.metrics = LayerMetrics(self.macs, v)


    @abstractmethod
    def process(self, layer:Union[TfliteLayer, KerasLayer]):
        pass
    
    
def flat_size(shape : list):
    out = 1
    for k in shape:
        out *= k
        
    return out