# Model Quantization

This describes how to quantize an ML model using the MLTK's quantization command/API.

## Quick Reference

- Command-line: `mltk quantize --help`
- Python API: [quantize_model](mltk.core.quantize_model)
- Python API examples: [quantize_model.ipynb](../../mltk/examples/quantize_model.ipynb)


## Overview

Model quantization involves converting a model's float32 weights and filters to an int8 representation.
Quantizing a model can reduce flash and RAM usage by 4x.

Model quantization is performed using the [Tensorflow-Lite Converter](https://www.tensorflow.org/lite/convert).  
Refer to [Post-training Quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)
for more details about how quantization is implemented.

Model quantization happens __automatically__ at the end of [model training](./model_training.md).  
The output of model quantization is a `.tflite` model file that can be directly programmed to an
embedded device and executed by the [Tensorflow-Lite Micro](https://github.com/tensorflow/tflite-micro) interpreter.

Additionally, model quantization can be invoked via the `quantize` command or the [quantize_model](mltk.core.quantize_model) API,
either of these will also generate a `.tflite` model file.

```{note}
When the `.tflite` model file is generated, model parameters are also added to the file's "metadata" section.
See [Model Parameters](./model_parameters.md) for more details.
```


## Tensorflow-Lite Converter Settings

Model quantization is performed using the [Tensorflow-Lite Converter](https://www.tensorflow.org/lite/convert).
The settings for the converter are configured in the [model specification](./model_specification.md) script
using the model property: [TrainMixin.tflite_converter](mltk.core.TrainMixin.tflite_converter).

For example, the model specification script might have:

```python
my_model.tflite_converter['optimizations'] = [tf.lite.Optimize.DEFAULT]
my_model.tflite_converter['supported_ops'] = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
my_model.tflite_converter['inference_input_type'] = tf.int8
my_model.tflite_converter['inference_output_type'] = tf.int8
my_model.tflite_converter['representative_dataset'] = 'generate'
```

```{seealso}
- [Post-training Quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)
- [TFLiteConverter](https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter)
```


The following parameters are used by the [TFLiteConverter](https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter):


### optimizations

This is a set of optimizations to apply.   
Currently, this should always be set to `[tf.lite.Optimize.DEFAULT]`.

### supported_ops

This sets the `target_spec` which configures the  `supported_ops` field of [tf.lite.TargetSpec](https://www.tensorflow.org/api_docs/python/tf/lite/TargetSpec).

Currently, this should always be set to `[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]`.

### inference_input_type

Data type of the model input layer. Note that integer types (`tf.int8` and `tf.uint8`) are currently only supported for post training integer quantization.  
(default `tf.float32`, must be in `{tf.float32, tf.int8, tf.uint8}`).

It's recommended to use `tf.int8`.  
If `tf.float32` then the converter will automatically add additional de-quantization/quantization layers 
to the `.tflite` model to convert to/from int8.


### inference_output_type

Data type of the model output layer. Note that integer types (`tf.int8` and `tf.uint8`) are currently only supported for post training integer quantization.  
(default `tf.float32`, must be in `{tf.float32, tf.int8, tf.uint8}`)

It's recommended to use `tf.int8`.  
If `tf.float32` then the converter will automatically add additional de-quantization/quantization layers 
to the `.tflite` model to convert to/from int8.


### representative_dataset

A generator function used for integer quantization where each generated sample has the same order, type and shape as the inputs to the model. Usually, this is a small subset of a few hundred samples randomly chosen, in no particular order, from the training or evaluation dataset.

If the keyword `generate` is used, then the MLTK will automatically create a generator function from the model's validation data.




## Command

Model quantization from the command-line is done using the `quantize` operation.  
This command should be invoked __after__ [model training](./model_training.md).

This command is useful as it allows for modifying a trained model's [TrainMixin.tflite_converter](mltk.core.TrainMixin.tflite_converter)
settings to generate new `.tflite` model file.


For more details on the available command-line options, issue the command:

```shell
mltk quantize --help
```


### Example 1: Update .tflite in model archive

The most common use case of the `quantize` command is:
1. Fully [train](./model_training.md) a model
2. Later modify the TrainMixin.tflite_converter](mltk.core.TrainMixin.tflite_converter) settings in the [model specification](./model_specification.md) script
3. Run the `quantize` command to update the `.tflite` model file in the [model archive](./model_archive.md)


In this example, it's assumed that the [TrainMixin.tflite_converter](mltk.core.TrainMixin.tflite_converter) settings in
the [image_classification](mltk.models.tinyml.image_classification) model specification script have been modified _after_ the model has been
trained.

```shell
mltk quantize image_classification
```

After this command completes, the `image_classification.mltk.zip` model archive is updated with a new `image_classification.tflite` model file.



## Python API

Model quantization is accessible via the [quantize_model](mltk.core.quantize_model) API.  
This API should be invoked __after__ [model training](./model_training.md).

This API is useful as it allows for modifying a trained model's [TrainMixin.tflite_converter](mltk.core.TrainMixin.tflite_converter)
settings to generate a new `.tflite` model file.

Examples using this API may be found in [quantize_model.ipynb](../../mltk/examples/quantize_model.ipynb)

