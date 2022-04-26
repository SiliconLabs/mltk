
from mltk.core.tflite_model import TfliteOpCode
from mltk.core.profiling_results import ProfilingLayerResult

from .base_estimator import BaseEstimator

_estimators = {
    'mvp': {
        TfliteOpCode.ADD: BaseEstimator('add', 'mvp'),
        TfliteOpCode.CONV_2D: BaseEstimator('conv2d', 'mvp'),
        TfliteOpCode.DEPTHWISE_CONV_2D: BaseEstimator('depthwise_conv2d', 'mvp'),
        TfliteOpCode.FULLY_CONNECTED: BaseEstimator('fully_connected', 'mvp'),
        TfliteOpCode.MAX_POOL_2D: BaseEstimator('max_pool2d', 'mvp'),
        TfliteOpCode.AVERAGE_POOL_2D: BaseEstimator('average_pool2d', 'mvp'),
    },
    'none': {
        TfliteOpCode.ADD: BaseEstimator('add'),
        TfliteOpCode.AVERAGE_POOL_2D: BaseEstimator('average_pool2d'),
        TfliteOpCode.CONV_2D: BaseEstimator('conv2d'),
        TfliteOpCode.DEPTHWISE_CONV_2D: BaseEstimator('depthwise_conv2d'),
        TfliteOpCode.DEQUANTIZE: BaseEstimator('dequantize'),
        TfliteOpCode.FULLY_CONNECTED: BaseEstimator('fully_connected'),
        TfliteOpCode.MAX_POOL_2D: BaseEstimator('max_pool_2d'),
        TfliteOpCode.MEAN: BaseEstimator('mean'),
        TfliteOpCode.PAD: BaseEstimator('pad'),
        TfliteOpCode.QUANTIZE: BaseEstimator('quantize'),
        TfliteOpCode.RESHAPE: BaseEstimator('reshape'),
        TfliteOpCode.SOFTMAX: BaseEstimator('softmax'),
    }
}

def get_estimates(
    accelerator:str,
    layer:ProfilingLayerResult,  
    cpu_clock_rate:int,
    **kwargs
):
    # Convert the accelerator arg to 'none' or 'mvp'
    accelerator = 'none' if accelerator is None else accelerator.lower()
    if accelerator not in _estimators:
        return

    estimators = _estimators[accelerator]

    # If the layer's accelerator cycles are 0 (i.e. it wasn't able to be accelerated by MVP)
    # or the kernel (i.e. opcode) is not supported by the MVP
    # then revert to the 'none' (i.e. CMSIS-only) estimators
    if layer.accelerator_cycles == 0 or layer.opcode not in estimators:
        estimators = _estimators['none']

    # If this kernel does not have an estimator
    # then just return
    if layer.opcode not in estimators:
        return

    # Update the given layer with the estimators predictions
    estimator = estimators[layer.opcode]
    estimator.predict(
        layer=layer, 
        cpu_clock_rate=cpu_clock_rate
    )
