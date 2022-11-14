import abc

from typing import Union

from mltk.utils.python import prepend_exception_msg
from mltk.core.tflite_model import TfliteLayer
from mltk.core.profiling_results import ProfilingLayerResult

from .utils import (
    MetricBaseEstimator,
    padding_to_int,
    activation_to_int, 
    use_activation, bool_to_int,
    load_model
)



class BaseEstimator(abc.ABC):

    def __init__(self, name, accelerator=None):
        self.accelerator = 'none' if accelerator is None else accelerator.lower()
        self.name = name.lower()
        self._loaded = False
        self.cpu_cycles_model:MetricBaseEstimator = None 
        self.energy_model:MetricBaseEstimator = None


    def load_models(self):
        if self._loaded:
            return 

        self._loaded = True
        self.cpu_cycles_model = load_model(
            name=self.name, 
            accelerator=self.accelerator,
            metric='cpu_cycles'
        )
        self.energy_model = load_model(
            name=self.name, 
            accelerator=self.accelerator,
            metric='energy'
        )

    def predict(
        self, 
        layer: ProfilingLayerResult,  
        cpu_clock_rate:int
    ):
        # pylint: disable=protected-access
        self.load_models()

        if self.cpu_cycles_model is not None:
            layer['cpu_cycles'] = self.predict_cpu_cycles(layer, layer.accelerator_cycles)

        if self.energy_model is not None:
            layer['energy'] = self.predict_energy(
                layer, 
                accelerator_cycles=layer.accelerator_cycles,
                cpu_cycles=layer.cpu_cycles,
            )

        layer._time = self.predict_time(
            accelerator_cycles=layer.accelerator_cycles, 
            cpu_cycles=layer.cpu_cycles,
            cpu_clock_rate=cpu_clock_rate
        )


    def predict_time(
        self, 
        accelerator_cycles:int, 
        cpu_cycles:int, 
        cpu_clock_rate: int, 
    ) -> float:
        # We assume the time spent processing the layer is
        # the maximum of the accelerator or CPU cycles
        # over the clock rate
        return max(accelerator_cycles, cpu_cycles) / cpu_clock_rate


    def predict_cpu_cycles(
        self, 
        layer: ProfilingLayerResult, 
        accelerator_cycles:int
    ) -> int:
        return predict_layer_value(
            self.cpu_cycles_model, 
            layer, 
            accelerator_cycles=accelerator_cycles
        )


    def predict_energy(
        self, 
        layer: ProfilingLayerResult, 
        accelerator_cycles:int,
        cpu_cycles:int
    ) -> int:
        return predict_layer_value(
            self.energy_model, 
            layer, 
            accelerator_cycles=accelerator_cycles,
            cpu_cycles=cpu_cycles
        )





def predict_layer_value(
    model, 
    layer: Union[ProfilingLayerResult,TfliteLayer], 
    accelerator_cycles:int=None,
    cpu_cycles:int=None
):
    params = extract_model_parameters(
        model, 
        layer, 
        accelerator_cycles=accelerator_cycles,
        cpu_cycles=cpu_cycles
    )
    return model.predict(**params)


def extract_model_parameters(
    model, 
    layer: Union[ProfilingLayerResult,TfliteLayer], 
    accelerator_cycles:int=None,
    cpu_cycles:int=None,
) -> dict:
    if isinstance(model, list):
        feature_names = model
    else:
        feature_names = model.feature_names
    params = {}

    if isinstance(layer, ProfilingLayerResult):
        tflite_layer = layer.tflite_layer
    else:
        tflite_layer = layer 
    
    layer_input = tflite_layer.inputs[0]
    layer_output = tflite_layer.outputs[0]

    input_shape_base = 0 if len(layer_input.shape) == 2 else 1
    output_shape_base = 0 if len(layer_input.shape) == 2 else 1

    try:
        for name in feature_names:
            if name == 'accelerator_cycles':
                assert accelerator_cycles is not None and accelerator_cycles > 0
                params['accelerator_cycles'] = accelerator_cycles
            elif name == 'cpu_cycles':
                assert cpu_cycles is not None and cpu_cycles > 0
                params['cpu_cycles'] = cpu_cycles
            elif name == 'accelerator_loads':
                params['accelerator_loads'] = layer['accelerator_loads']
            elif name == 'accelerator_optimized_loads':
                params['accelerator_optimized_loads'] = layer['accelerator_optimized_loads']
            elif name == 'accelerator_parallel_loads':
                params['accelerator_parallel_loads'] = layer['accelerator_parallel_loads']
            elif name == 'input_size':
                params['input_size'] = layer_input.shape.flat_size
            elif name == 'input_flat_size':
                params['input_flat_size'] = layer_input.shape.flat_size
            elif name == 'output_size': 
                params['output_size'] = layer_output.shape.flat_size
            elif name == 'output_flat_size': 
                params['output_flat_size'] = layer_output.shape.flat_size
            elif name == 'input_height':
                params['input_height'] = layer_input.shape[input_shape_base+0]
            elif name == 'input_width':
                params['input_width'] = layer_input.shape[input_shape_base+1]
            elif name == 'input_depth':
                params['input_depth'] = 1 if len(layer_input.shape) != 4 else layer_input.shape[3] 
            elif name == 'output_height':
                params['output_height'] = layer_output.shape[output_shape_base+0]
            elif name == 'output_width':
                params['output_width'] = layer_output.shape[output_shape_base+1]
            elif name == 'output_depth':
                params['output_depth'] = 1 if len(layer_output.shape) != 4 else layer_output.shape[3] 
            elif name == 'stride_height':
                params['stride_height'] = tflite_layer.strides[0]
            elif name == 'stride_width':
                params['stride_width'] = tflite_layer.strides[1]
            elif name == 'kernel_height':
                params['kernel_height'] = tflite_layer.kernel_size[0]
            elif name == 'kernel_width':
                params['kernel_width'] = tflite_layer.kernel_size[1]
            elif name == 'pool_height':
                params['pool_height'] = tflite_layer.pool_size[0]
            elif name == 'pool_width':
                params['pool_width'] = tflite_layer.pool_size[1]
            elif name == 'filters':
                params['filters'] = tflite_layer.filters
            elif name == 'multiplier':
                params['multiplier'] = tflite_layer.multiplier
            elif name == 'padding':
                params['padding'] = padding_to_int(tflite_layer.padding)
            elif name == 'activation':
                params['activation'] = activation_to_int(tflite_layer.activation)
            elif name == 'use_activation':
                params['use_activation'] = use_activation(tflite_layer.activation)
            elif name == 'use_bias':
                params['use_bias'] = bool_to_int(tflite_layer.use_bias)
            elif name == 'requires_copy':
                params['requires_copy'] = bool_to_int(tflite_layer.requires_copy)
            elif name == 'n_elements':
                params['n_elements'] = tflite_layer.n_input_elements
            elif name == 'accumulator_depth':
                params['accumulator_depth'] = tflite_layer.accumulator_depth
            elif name == 'use_parallel_mac':
                params['use_parallel_mac'] = bool_to_int(tflite_layer.accumulator_depth % 2 == 0)
            elif name == 'parallelize_channels':
                params['parallelize_channels'] = bool_to_int(tflite_layer.mutliplier % 2 == 0)
            elif name == 'units':
                params['units'] = tflite_layer.units
            else:
                raise Exception(f'Unknown feature: {name}')
    except Exception as e:
        prepend_exception_msg(e, f'Layer: {layer.name}, failed to extract feature: {name}')
        raise

    return params
