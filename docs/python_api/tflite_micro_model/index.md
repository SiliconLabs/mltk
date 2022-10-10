# Tensorflow-Lite Micro Model

This package allows for executing a quantized `.tflite` model file in the [Tensorflow-Lite Micro](https://github.com/tensorflow/tflite-micro) interpreter.

Example usage of this package is as follows:

```python
from mltk.core.tflite_micro import TfliteMicro

# Profile the model in the TFLM interpreter
profiling_results = TfliteMicro.profile_model(tflite_path)

print(profiling_results)
```

See the [TfliteMicroModel API examples](https://siliconlabs.github.io/mltk/mltk/examples/tflite_micro_model.html) for more examples.




```{eval-rst}
.. autosummary::
   :toctree: model
   :template: custom-class-template.rst

   mltk.core.tflite_micro.TfliteMicroModel

.. autosummary::
   :toctree: model_details
   :template: custom-class-template.rst

   mltk.core.tflite_micro.TfliteMicroModelDetails

.. autosummary::
   :toctree: wrapper
   :template: custom-class-template.rst

   mltk.core.tflite_micro.TfliteMicro

.. autosummary::
   :toctree: profiled_layer_result
   :template: custom-class-template.rst

   mltk.core.tflite_micro.TfliteMicroProfiledLayerResult

.. autosummary::
   :toctree: recorded_layer_result
   :template: custom-class-template.rst

   mltk.core.tflite_micro.TfliteMicroRecordedLayerResult

.. autosummary::
   :toctree: layer_error
   :template: custom-class-template.rst

   mltk.core.tflite_micro.TfliteMicroLayerError

.. autosummary::
   :toctree: accelerator
   :template: custom-class-template.rst

   mltk.core.tflite_micro.tflite_micro_accelerator.TfliteMicroAccelerator

```


```{toctree}
:maxdepth: 1
:hidden:

./model
./model_details
./wrapper
./accelerator
./layer_error
./recorded_layer_result
./profiled_layer_result
```