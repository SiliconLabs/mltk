from typing import Union
import logging

from tensorflow.keras.layers import Layer as KerasLayer
from tensorflow.keras.models import Model as KerasModel 

from mltk.core import (TfliteLayer, TfliteModel)
from mltk.utils.logger import DummyLogger

from .layers import load_layers, parse_layer
from .layers.layer import LayerMetrics


def calculate_model_metrics(
    model: Union[TfliteModel, KerasModel], 
    logger:logging.Logger=None
) -> dict:
    """Calculate the MACs/OPs required by given KerasModel or TfliteModel model"""
    load_layers() 

    if not isinstance(model, (TfliteModel, KerasModel)):
        raise Exception('Model must be an instance of a TfliteModel or KerasModel')

    logger = logger or DummyLogger()

    layers_metrics = []


    def _process_model_layers(model):
        for model_layer in model.layers:
            if isinstance(model_layer, KerasModel):
                _process_model_layers(model_layer)
                continue

            layer = parse_layer(model_layer)
            if layer is None:
                logger.debug(f'WARN: Model layer not included in metrics calculation, name: {model_layer.name}, type: {model_layer.__class__.__name__}')
                continue 
            layers_metrics.append(layer)

    _process_model_layers(model)
    total_macs = sum(map(lambda x: x.macs, layers_metrics))
    total_ops = sum(map(lambda x: x.ops, layers_metrics))

    return dict(
        total_macs=total_macs,
        total_ops=total_ops,
        layers=layers_metrics
    )


def calculate_layer_metrics(layer: Union[TfliteLayer, KerasLayer]) -> LayerMetrics:
    """Calculate the metrics for the given KerasLayer or TfliteLayer
    Return LayerMetrics(0,0) if the layer is not supported
    """
    load_layers() 
    
    layer_metrics = parse_layer(layer)
    if layer_metrics is None:
        return LayerMetrics(0, 0)
    return layer_metrics.metrics
