# Model Training


This describes how to train a Machine Learning model using the MLTK and [Google Tensorflow](https://www.tensorflow.org).

```{note}
This document focuses on the training aspect of model development.
Refer to the [tutorials](../tutorials.md) for end-to-end guides on how to develop an ML model.
```



## Quick Reference

- Command-line: `mltk train --help`
- Python API: [train_model](mltk.core.train_model)
- Python API examples: [train_model.ipynb](../../mltk/examples/train_model.ipynb)


## Overview

The MLTK internally uses [Google Tensorflow](https://www.tensorflow.org/tutorials) to train a model.  
The basic sequence for training a model is:

1. Create a [model specification](./model_specification.md) script
2. Populate the model training and dataset parameters
3. Define the model layout using the [Keras API](https://keras.io/api)
4. Invoke model training using the [Command-Line](#command) or [Python API](#python-api)

When training completes, a [model archive](./model_archive.md) file is generated in the same directory as the model training script and contains the trained model files and logs.

__HINT:__ See [Training via SSH](./model_training_via_ssh.md) for how to quickly train your model in the cloud.

## Model Specification

All model training parameters are defined in the [Model Specification](./model_specification.md) script.
This is a standard Python script that defines a [MltkModel](mltk.core.MltkModel) instance.

### MltkModel Instance

All training parameters are configured in the [MltkModel](mltk.core.MltkModel) instance.  
For example, the following might be added to the top of `my_model_v1.py`:

```python
# Define a new MyModel class which inherits the 
# MltkModel and several mixins
# @mltk_model
class MyModel(
    MltkModel, 
    TrainMixin, 
    AudioDatasetMixin, 
    EvaluateClassifierMixin
):
    """My Model's class object"""

# Instantiate the MyModel class
my_model = MyModel()
```

Here we define our model's class object: `MyModel`.

At a minimum, this custom class must inherit the following:  
- [MltkModel](mltk.core.MltkModel)
- [TrainMixin](mltk.core.TrainMixin)
- [DatasetMixin](mltk.core.DatasetMixin) (or a child of this mixin)

Additionally, this class inherits other model "mixins" to aid model development.

After our model is instantiated, the rest of the model specification simply 
populates the various properties of `MyModel`, e.g.:  

```python
# General Settings
my_model.version = 1
my_model.description = 'My model is great!'

# Training Basic Settings
my_model.epochs = 100
my_model.batch_size = 64 
my_model.optimizer = 'adam'
...

# Dataset Settings
my_model.dataset = speech_commands_v2
my_model.class_mode = 'categorical'
my_model.classes = ['up', 'down', 'left', 'right']
...
```

```{note}
The filename of the model specification script is the name given to the model. So, in this case, the model name is `my_model_v1`.
```

### Model Layout

An important property of the `MyModel` class example from above is
[TrainMixin.build_model_function](mltk.core.TrainMixin.build_model_function). This should reference
a function that builds the actual machine learning model which is built using the [Keras API](https://keras.io/api).

For example:

```python
def my_model_builder(my_model: MyModel):
    keras_model = Sequential(name=my_model.name)

    keras_model = Sequential()
    keras_model.add(InputLayer(my_model.input_shape))
    keras_model.add(Conv2D(
        filters=8,
        kernel_size=(10, 8),
        use_bias=True,
        padding="same",
        strides=(2,2))
    )
    keras_model.add(Activation('relu'))
    keras_model.add(Flatten())
    keras_model.add(Dense(units=my_model.n_classes))

    keras_model.compile(
        loss=my_model.loss, 
        optimizer=my_model.optimizer, 
        metrics=my_model.metrics
    )
    return keras_model

# Set the model property to reference the model build function
my_model.build_model_function = my_model_builder
```

Here, we define a function that builds a [KerasModel](mltk.core.KerasModel) then
sets `my_model.build_model_function` to reference the function.

At model training time, the model building function is invoked and the built 
[KerasModel](mltk.core.KerasModel) is trained using Tensorflow.


#### Note about hardcoding model layer parameters

While not required, the `my_model` argument to the building function
should be used over hardcoded values, e.g.:

```python
# Good:
# Dynamically determine the number of dense unit based
# on the number of classes specified in the model properties
keras_model.add(Dense(units=my_model.n_classes))

# Bad:
# Hardcoding dense units
# If the number of classes changes, 
# then training will likely fail
keras_model.add(Dense(units=5))
```

See the [Model Specification](./model_specification.md) documentation for more details.


## Training Output

When training completes, a [model archive](./model_archive.md) file is generated in the same directory as the model specification script and contains the trained model files and logs.

Included in the [model archive](./model_archive.md) is a quantized, `.tflite` model file. 
This is the file that is programmed into the embedded device and executed by [Tensorflow-Lite Micro](https://github.com/tensorflow/tflite-micro).

The `.tflite` is generated by the Tensorflow-Lite [Converter](https://www.tensorflow.org/lite/convert).
The settings for the converter are defined in the model specification script using the model property: [TrainMixin.tflite_converter](mltk.core.TrainMixin.tflite_converter)


For example, the model specification script might have:

```python
my_model.tflite_converter['optimizations'] = [tf.lite.Optimize.DEFAULT]
my_model.tflite_converter['supported_ops'] = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
my_model.tflite_converter['inference_input_type'] = tf.int8
my_model.tflite_converter['inference_output_type'] = tf.int8
my_model.tflite_converter['representative_dataset'] = 'generate'
```
These settings are used at the end of training to generate the `.tflite`.  
See [Model Quantization](./model_quantization.md) for more details.


## Command

Model training from the command-line is done using the `train` operation.

For more details on the available command-line options, issue the command:

```shell
mltk train --help
```

__HINT:__ See [Training via SSH](./model_training_via_ssh.md) for how to quickly train your model in the cloud.

The following are examples of how training can be invoked from the command-line:

### Example 1: Train as a "dry run"

Before fully training a model, sometimes it is useful to do a "dry run" 
to ensure everything is working. This can be done by appending `-test` to the end 
of the model name. With this, the model is trained for 1 epoch on a subset of 
the training data, and a model archive with `-test` append to the name is generated. 

```shell
mltk train tflite_micro_speech-test
```

### Example 2: Train for 100 epochs

The model specification typically contains the number of training epochs, i.e. [TrainMixin.epochs](mltk.core.TrainMixin.epochs).  
Optionally, the `--epochs` option can be used to override the model specification.

```shell
mltk train audio_example1 --epochs 100
```

### Example 3: Resume Training

If training does not fully complete, it can be restarted by adding the `--resume` option.
This will load the weights from the last saved checkpoint and begin training at that checkpoint's epoch.
See [TrainMixin.checkpoint](mltk.core.TrainMixin.checkpoint) for more details.

```shell
mltk train image_example1 --resume
```

## Python API

Model training is accessible via [train_model](mltk.core.train_model) API.

Examples using this API may be found in [train_model.ipynb](../../mltk/examples/train_model.ipynb)

