__NOTE:__ Refer to the [online documentation](https://siliconlabs.github.io/mltk) to properly view this file
# Tensorflow-Lite Model

This allows for accessing [.tflite](https://www.tensorflow.org/lite/models/convert) model files.
A `.tflite` uses a binary format called a [flatbuffer](https://google.github.io/flatbuffers/).
The flatbuffer "schema" used by a `.tflite` model is defined in [schema.fbs](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs).


## Example Usage

Example usage of this package is as follows:

```python
# Import the TfliteModel class
from mltk.core import TfliteModel

# Load the .tflite
tflite_model = TfliteModel.load_flatbuffer_file(tflite_path)

# Generate a summary of the .tflite
summary = tflite_model.summary()

# Print the summary to the console
print(summary)
```

See the [TfliteModel API examples](https://siliconlabs.github.io/mltk/mltk/examples/tflite_model.html) for more examples.

## API Reference


```{eval-rst}
.. autosummary::
   :toctree: model
   :template: custom-class-template.rst

   mltk.core.TfliteModel

.. autosummary::
   :toctree: layer
   :template: custom-class-template.rst

   mltk.core.TfliteLayer

.. autosummary::
   :toctree: add_layer
   :template: custom-class-template.rst

   mltk.core.TfliteAddLayer

.. autosummary::
   :toctree: conv2d_layer
   :template: custom-class-template.rst

   mltk.core.TfliteConv2dLayer

.. autosummary::
   :toctree: conv2d_layer_options
   :template: custom-class-template.rst

   mltk.core.TfliteConv2DLayerOptions

.. autosummary::
   :toctree: conv_params
   :template: custom-class-template.rst

   mltk.core.TfliteConvParams

.. autosummary::
   :toctree: transpose_conv_layer
   :template: custom-class-template.rst

   mltk.core.TfliteTransposeConvLayer

.. autosummary::
   :toctree: transpose_conv_layer_options
   :template: custom-class-template.rst

   mltk.core.TfliteTransposeConvLayerOptions

.. autosummary::
   :toctree: transpose_conv_params
   :template: custom-class-template.rst

   mltk.core.TfliteTransposeConvParams

.. autosummary::
   :toctree: fully_connected_layer
   :template: custom-class-template.rst

   mltk.core.TfliteFullyConnectedLayer

.. autosummary::
   :toctree: fully_connected_layer_options
   :template: custom-class-template.rst

   mltk.core.TfliteFullyConnectedLayerOptions

.. autosummary::
   :toctree: fully_connected_params
   :template: custom-class-template.rst

   mltk.core.TfliteFullyConnectedParams

.. autosummary::
   :toctree: depthwise_conv2d_layer
   :template: custom-class-template.rst

   mltk.core.TfliteDepthwiseConv2dLayer

.. autosummary::
   :toctree: depthwise_conv2d_layer_options
   :template: custom-class-template.rst

   mltk.core.TfliteDepthwiseConv2DLayerOptions

.. autosummary::
   :toctree: depthwise_conv_params
   :template: custom-class-template.rst

   mltk.core.TfliteDepthwiseConvParams

.. autosummary::
   :toctree: pooling2d_layer
   :template: custom-class-template.rst

   mltk.core.TflitePooling2dLayer

.. autosummary::
   :toctree: pooling2d_layer_options
   :template: custom-class-template.rst

   mltk.core.TflitePool2DLayerOptions

.. autosummary::
   :toctree: pooling_params
   :template: custom-class-template.rst

   mltk.core.TflitePoolParams

.. autosummary::
   :toctree: reshape_layer
   :template: custom-class-template.rst

   mltk.core.TfliteReshapeLayer

.. autosummary::
   :toctree: quantize_layer
   :template: custom-class-template.rst

   mltk.core.TfliteQuantizeLayer

.. autosummary::
   :toctree: dequantize_layer
   :template: custom-class-template.rst

   mltk.core.TfliteDequantizeLayer

.. autosummary::
   :toctree: unidirectional_sequence_lstm_layer
   :template: custom-class-template.rst

   mltk.core.TfliteUnidirectionalLstmLayer

.. autosummary::
   :toctree: unidirectional_sequence_lstm_layer_options
   :template: custom-class-template.rst

   mltk.core.TfliteUnidirectionalLstmLayerOptions

.. autosummary::
   :toctree: tensor
   :template: custom-class-template.rst

   mltk.core.TfliteTensor

.. autosummary::
   :toctree: shape
   :template: custom-class-template.rst

   mltk.core.TfliteShape

.. autosummary::
   :toctree: quantization
   :template: custom-class-template.rst

   mltk.core.TfliteQuantization

.. autosummary::
   :toctree: activation
   :template: custom-class-template.rst

   mltk.core.TfliteActivation

.. autosummary::
   :toctree: padding
   :template: custom-class-template.rst

   mltk.core.TflitePadding

.. autosummary::
   :toctree: parameters
   :template: custom-class-template.rst

   mltk.core.TfliteModelParameters
```


```{toctree}
:maxdepth: 1
:hidden:

./model
./parameters
./dictionary.fbs.md
./layer
./add_layer
./conv2d_layer
./conv2d_layer_options
./conv_params
./transpose_conv_layer
./transpose_conv_layer_options
./transpose_conv_params
./fully_connected_layer
./fully_connected_layer_options
./fully_connected_params
./depthwise_conv2d_layer
./depthwise_conv2d_layer_options
./depthwise_conv_params
./pooling2d_layer
./pooling2d_layer_options
./pooling_params
./reshape_layer
./quantize_layer
./dequantize_layer
./unidirectional_sequence_lstm_layer
./unidirectional_sequence_lstm_layer_options
./tensor
./shape
./quantization
./activation
./padding
```