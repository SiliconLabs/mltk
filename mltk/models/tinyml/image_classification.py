"""image_classification
*************************

Image classification (ResNetv1-10 with CIFAR10)

- Source code: `image_classification.py <https://github.com/siliconlabs/mltk/blob/master/mltk/models/tinyml/image_classification.py>`_
- Pre-trained model: `image_classification.mltk.zip <https://github.com/siliconlabs/mltk/blob/master/mltk/models/tinyml/image_classification.mltk.zip>`_


This was adapted from:

* https://github.com/mlcommons/tiny/tree/master/benchmark/training/image_classification
* https://github.com/SiliconLabs/platform_ml_models/blob/master/eembc/CIFAR10_ResNetv1/cifar10_main.py


Dataset
----------
* https://www.cs.toronto.edu/~kriz/cifar.html

Model Topology
--------------
* https://arxiv.org/pdf/1512.03385.pdf
* https://keras.io/api/applications/resnet/

Performance (floating point model)
----------------------------------
* Accuracy - 86.2%
* AUC - .989

Performance (quantized tflite model)
------------------------------------
* Accuracy - 86.1%
* AUC - .988


Commands
--------------

.. code-block:: shell

   # Do a "dry run" test training of the model
   mltk train image_classification-test

   # Train the model
   mltk train image_classification

   # Evaluate the trained model .tflite model
   mltk evaluate image_classification --tflite

   # Profile the model in the MVP hardware accelerator simulator
   mltk profile image_classification --accelerator MVP

   # Profile the model on a physical development board
   mltk profile image_classification --accelerator MVP --device


Model Summary
--------------

.. code-block:: shell
    
    mltk summarize image_classification --tflite
    
    +-------+-----------------+-----------------+-----------------+-----------------------------------------------------+
    | Index | OpCode          | Input(s)        | Output(s)       | Config                                              |
    +-------+-----------------+-----------------+-----------------+-----------------------------------------------------+
    | 0     | conv_2d         | 32x32x3 (int8)  | 32x32x16 (int8) | Padding:same stride:1x1 activation:relu             |
    |       |                 | 3x3x3 (int8)    |                 |                                                     |
    |       |                 | 16 (int32)      |                 |                                                     |
    | 1     | conv_2d         | 32x32x16 (int8) | 32x32x16 (int8) | Padding:same stride:1x1 activation:relu             |
    |       |                 | 3x3x16 (int8)   |                 |                                                     |
    |       |                 | 16 (int32)      |                 |                                                     |
    | 2     | conv_2d         | 32x32x16 (int8) | 32x32x16 (int8) | Padding:same stride:1x1 activation:none             |
    |       |                 | 3x3x16 (int8)   |                 |                                                     |
    |       |                 | 16 (int32)      |                 |                                                     |
    | 3     | add             | 32x32x16 (int8) | 32x32x16 (int8) | Activation:relu                                     |
    |       |                 | 32x32x16 (int8) |                 |                                                     |
    | 4     | conv_2d         | 32x32x16 (int8) | 16x16x32 (int8) | Padding:same stride:2x2 activation:relu             |
    |       |                 | 3x3x16 (int8)   |                 |                                                     |
    |       |                 | 32 (int32)      |                 |                                                     |
    | 5     | conv_2d         | 16x16x32 (int8) | 16x16x32 (int8) | Padding:same stride:1x1 activation:none             |
    |       |                 | 3x3x32 (int8)   |                 |                                                     |
    |       |                 | 32 (int32)      |                 |                                                     |
    | 6     | conv_2d         | 32x32x16 (int8) | 16x16x32 (int8) | Padding:same stride:2x2 activation:none             |
    |       |                 | 1x1x16 (int8)   |                 |                                                     |
    |       |                 | 32 (int32)      |                 |                                                     |
    | 7     | add             | 16x16x32 (int8) | 16x16x32 (int8) | Activation:relu                                     |
    |       |                 | 16x16x32 (int8) |                 |                                                     |
    | 8     | conv_2d         | 16x16x32 (int8) | 8x8x64 (int8)   | Padding:same stride:2x2 activation:relu             |
    |       |                 | 3x3x32 (int8)   |                 |                                                     |
    |       |                 | 64 (int32)      |                 |                                                     |
    | 9     | conv_2d         | 8x8x64 (int8)   | 8x8x64 (int8)   | Padding:same stride:1x1 activation:none             |
    |       |                 | 3x3x64 (int8)   |                 |                                                     |
    |       |                 | 64 (int32)      |                 |                                                     |
    | 10    | conv_2d         | 16x16x32 (int8) | 8x8x64 (int8)   | Padding:same stride:2x2 activation:none             |
    |       |                 | 1x1x32 (int8)   |                 |                                                     |
    |       |                 | 64 (int32)      |                 |                                                     |
    | 11    | add             | 8x8x64 (int8)   | 8x8x64 (int8)   | Activation:relu                                     |
    |       |                 | 8x8x64 (int8)   |                 |                                                     |
    | 12    | average_pool_2d | 8x8x64 (int8)   | 1x1x64 (int8)   | Padding:valid stride:8x8 filter:8x8 activation:none |
    | 13    | reshape         | 1x1x64 (int8)   | 64 (int8)       | BuiltinOptionsType=0                                |
    |       |                 | 2 (int32)       |                 |                                                     |
    | 14    | fully_connected | 64 (int8)       | 10 (int8)       | Activation:none                                     |
    |       |                 | 64 (int8)       |                 |                                                     |
    |       |                 | 10 (int32)      |                 |                                                     |
    | 15    | softmax         | 10 (int8)       | 10 (int8)       | BuiltinOptionsType=9                                |
    +-------+-----------------+-----------------+-----------------+-----------------------------------------------------+
    Total MACs: 12.502 M
    Total OPs: 25.122 M
    Name: image_classification
    Version: 1
    Description: TinyML: Image classification - ResNetv1-10 with CIFAR10
    Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    hash: d170adc21388920caa3f64ba22cd5b5d
    date: 2022-02-04T19:20:20.062Z
    runtime_memory_size: 53972
    samplewise_norm.rescale: 0.0
    samplewise_norm.mean_and_std: False
    .tflite file size: 99.3kB


Model Profiling Report
-----------------------

.. code-block:: shell
   
   # Profile on physical EFR32xG24 using MVP accelerator
   mltk profile image_classification --device --accelerator MVP

    Profiling Summary
    Name: image_classification
    Accelerator: MVP
    Input Shape: 1x32x32x3
    Input Data Type: int8
    Output Shape: 1x10
    Output Data Type: int8
    Flash, Model File Size (bytes): 99.2k
    RAM, Runtime Memory Size (bytes): 86.9k
    Operation Count: 25.2M
    Multiply-Accumulate Count: 12.4M
    Layer Count: 16
    Unsupported Layer Count: 0
    Accelerator Cycle Count: 9.3M
    CPU Cycle Count: 225.4k
    CPU Utilization (%): 2.4
    Clock Rate (hz): 78.0M
    Time (s): 118.9m
    Ops/s: 211.8M
    MACs/s: 104.7M
    Inference/s: 8.4

    Model Layers
    +-------+-----------------+--------+--------+------------+------------+----------+-------------------------+--------------+-----------------------------------------------------+
    | Index | OpCode          | # Ops  | # MACs | Acc Cycles | CPU Cycles | Time (s) | Input Shape             | Output Shape | Options                                             |
    +-------+-----------------+--------+--------+------------+------------+----------+-------------------------+--------------+-----------------------------------------------------+
    | 0     | conv_2d         | 933.9k | 442.4k | 583.4k     | 30.6k      | 7.5m     | 1x32x32x3,16x3x3x3,16   | 1x32x32x16   | Padding:same stride:1x1 activation:relu             |
    | 1     | conv_2d         | 4.8M   | 2.4M   | 1.8M       | 30.1k      | 22.3m    | 1x32x32x16,16x3x3x16,16 | 1x32x32x16   | Padding:same stride:1x1 activation:relu             |
    | 2     | conv_2d         | 4.7M   | 2.4M   | 1.8M       | 30.0k      | 22.3m    | 1x32x32x16,16x3x3x16,16 | 1x32x32x16   | Padding:same stride:1x1 activation:none             |
    | 3     | add             | 16.4k  | 0      | 57.4k      | 9.2k       | 840.0u   | 1x32x32x16,1x32x32x16   | 1x32x32x16   | Activation:relu                                     |
    | 4     | conv_2d         | 2.4M   | 1.2M   | 883.8k     | 15.3k      | 11.2m    | 1x32x32x16,32x3x3x16,32 | 1x16x16x32   | Padding:same stride:2x2 activation:relu             |
    | 5     | conv_2d         | 4.7M   | 2.4M   | 1.6M       | 30.2k      | 21.0m    | 1x16x16x32,32x3x3x32,32 | 1x16x16x32   | Padding:same stride:1x1 activation:none             |
    | 6     | conv_2d         | 270.3k | 131.1k | 131.2k     | 5.3k       | 1.7m     | 1x32x32x16,32x1x1x16,32 | 1x16x16x32   | Padding:same stride:2x2 activation:none             |
    | 7     | add             | 8.2k   | 0      | 28.7k      | 5.3k       | 450.0u   | 1x16x16x32,1x16x16x32   | 1x16x16x32   | Activation:relu                                     |
    | 8     | conv_2d         | 2.4M   | 1.2M   | 830.3k     | 15.3k      | 10.5m    | 1x16x16x32,64x3x3x32,64 | 1x8x8x64     | Padding:same stride:2x2 activation:relu             |
    | 9     | conv_2d         | 4.7M   | 2.4M   | 1.5M       | 30.2k      | 19.1m    | 1x8x8x64,64x3x3x64,64   | 1x8x8x64     | Padding:same stride:1x1 activation:none             |
    | 10    | conv_2d         | 266.2k | 131.1k | 115.4k     | 5.3k       | 1.5m     | 1x16x16x32,64x1x1x32,64 | 1x8x8x64     | Padding:same stride:2x2 activation:none             |
    | 11    | add             | 4.1k   | 0      | 14.4k      | 3.4k       | 240.0u   | 1x8x8x64,1x8x8x64       | 1x8x8x64     | Activation:relu                                     |
    | 12    | average_pool_2d | 4.2k   | 0      | 2.2k       | 3.9k       | 60.0u    | 1x8x8x64                | 1x1x1x64     | Padding:valid stride:8x8 filter:8x8 activation:none |
    | 13    | reshape         | 0      | 0      | 0          | 769.0      | 30.0u    | 1x1x1x64,2              | 1x64         | Type=none                                           |
    | 14    | fully_connected | 1.3k   | 640.0  | 1.0k       | 2.1k       | 30.0u    | 1x64,10x64,10           | 1x10         | Activation:none                                     |
    | 15    | softmax         | 50.0   | 0      | 0          | 8.2k       | 120.0u   | 1x10                    | 1x10         | Type=softmaxoptions                                 |
    +-------+-----------------+--------+--------+------------+------------+----------+-------------------------+--------------+-----------------------------------------------------+


Model Diagram
------------------

.. code-block:: shell
   
   mltk view image_classification --tflite

.. raw:: html

    <div class="model-diagram">
        <a href="../../../../_images/models/image_classification.tflite.png" target="_blank">
            <img src="../../../../_images/models/image_classification.tflite.png" />
            <p>Click to enlarge</p>
        </a>
    </div>


"""

import functools
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from mltk.core.preprocess.image.parallel_generator import ParallelImageDataGenerator
from mltk.core.model import (
    MltkModel,
    TrainMixin,
    ImageDatasetMixin,
    EvaluateClassifierMixin
)
from mltk.models.shared import ResNet10V1
from mltk.datasets.image import cifar10


# Instantiate the MltkModel object with the following 'mixins':
# - TrainMixin            - Provides classifier model training operations and settings
# - ImageDatasetMixin     - Provides image data generation operations and settings
# - EvaluateClassifierMixin         - Provides classifier evaluation operations and settings
# @mltk_model   # NOTE: This tag is required for this model be discoverable
class MyModel(
    MltkModel, 
    TrainMixin, 
    ImageDatasetMixin, 
    EvaluateClassifierMixin
):
    pass
my_model = MyModel()




# General parameters
my_model.version = 1
my_model.description = 'TinyML: Image classification - ResNetv1-10 with CIFAR10'


#################################################
# Training parameters
my_model.epochs = 200
my_model.batch_size = 40 
my_model.optimizer = 'adam'
my_model.metrics = ['accuracy']
my_model.loss = 'categorical_crossentropy'



#################################################
# TF-Lite converter settings
my_model.tflite_converter['optimizations'] = [tf.lite.Optimize.DEFAULT]
my_model.tflite_converter['supported_ops'] = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
my_model.tflite_converter['inference_input_type'] = tf.int8 # can also be tf.float32
my_model.tflite_converter['inference_output_type'] = tf.int8
 # generate a representative dataset from the validation data
my_model.tflite_converter['representative_dataset'] = 'generate'


#################################################
# Image Dataset Settings


# Default size for CIFAR10 dataset
input_height = 32
input_width = 32
input_depth = 3


# The classification type
my_model.class_mode = 'categorical'
# The class labels found in your training dataset directory
my_model.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# The input shape to the model. The dataset samples will be resized if necessary
my_model.input_shape = [input_height, input_width, input_depth]


def my_dataset_loader(model:MyModel):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # Convert for training
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Scale to INT8 range (simple non-adaptive)
    x_train = (x_train-128)/128
    x_test = (x_test-128)/128

    y_train = to_categorical(y_train, model.n_classes)
    y_test = to_categorical(y_test, model.n_classes)

    return  x_train, y_train, x_test, y_test


my_model.dataset = functools.partial(my_dataset_loader, my_model)



##############################################################
# Training callbacks
#

my_model.datagen = ParallelImageDataGenerator(
    cores=.35,
    max_batches_pending=32,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_augmentation_enabled=False
)



##############################################################
# Model Layout
def my_model_builder(model: MyModel):
    keras_model = ResNet10V1(
        input_shape=model.input_shape,
        num_classes=model.n_classes
    )
    keras_model.compile(
        loss=model.loss, 
        optimizer=model.optimizer, 
        metrics=model.metrics
    )
    return keras_model

my_model.build_model_function  = my_model_builder




##########################################################################################
# The following allows for running this model training script directly, e.g.: 
# python image_classification.py
#
# Note that this has the same functionality as:
# mltk train image_classification
#
if __name__ == '__main__':
    import mltk.core as mltk_core
    from mltk import cli

    # Setup the CLI logger
    cli.get_logger(verbose=False)

    # If this is true then this will do a "dry run" of the model testing
    # If this is false, then the model will be fully trained
    test_mode_enabled = True

    # Train the model
    # This does the same as issuing the command: mltk train image_classification-test --clean
    train_results = mltk_core.train_model(my_model, clean=True, test=test_mode_enabled)
    print(train_results)

    # Evaluate the model against the quantized .h5 (i.e. float32) model
    # This does the same as issuing the command: mltk evaluate image_classification-test
    tflite_eval_results = mltk_core.evaluate_model(my_model, verbose=True, test=test_mode_enabled)
    print(tflite_eval_results)

    # Profile the model in the simulator
    # This does the same as issuing the command: mltk profile image_classification-test
    profiling_results = mltk_core.profile_model(my_model, test=test_mode_enabled)
    print(profiling_results)