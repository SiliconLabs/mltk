"""visual_wake_words
**********************

Visual Wakeword - Person detection (MobileNetv1 with COCO14)

- Source code: `visual_wake_words.py <https://github.com/siliconlabs/mltk/blob/master/mltk/models/tinyml/visual_wake_words.py>`_
- Pre-trained model: `visual_wake_words.mltk.zip <https://github.com/siliconlabs/mltk/blob/master/mltk/models/tinyml/visual_wake_words.mltk.zip>`_

Taken from:

* https://github.com/mlcommons/tiny/blob/master/benchmark/training/visual_wake_words
* https://github.com/SiliconLabs/platform_ml_models/tree/master/eembc/Person_detection


Dataset
----------
* MSCOCO14 based [https://cocodataset.org/#download]
* Extraction based on COCO API [https://github.com/cocodataset/cocoapi]
* Person mimimal bounding box 2.5%
* 96x96 images resized with antialias filtering, no aspect ratio preservation
* All images converted to RGB
* Training and validation sets combined
* Dataset generation script (buildPersonDetectionDatabase.py) is included in repo
* Extracted Reference Dataset: `vw_coco2014_96.tar.gz <https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/vw_coco2014_96.tar.gz>`_

Model Topology
----------------
* Based on https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md
* Chosen configuration is a MobileNet_v1_0.25_96


Performance (floating point model) 
-----------------------------------
* Accuracy - 85.4%
* AUC - .931

Performance (quantized tflite model) 
------------------------------------
* Accuracy - 85.0%
* AUC - .928



Commands
--------------

.. code-block:: shell

   # Do a "dry run" test training of the model
   mltk train vww_model-test

   # Train the model
   mltk train vww_model

   # Evaluate the trained model .tflite model
   mltk evaluate vww_model --tflite

   # Profile the model in the MVP hardware accelerator simulator
   mltk profile vww_model --accelerator MVP

   # Profile the model on a physical development board
   mltk profile vww_model --accelerator MVP --device


Model Summary
--------------

.. code-block:: shell
    
    mltk summarize visual_wake_words --tflite
    
    +-------+-------------------+-------------------+-----------------+-----------------------------------------------------+
    | Index | OpCode            | Input(s)          | Output(s)       | Config                                              |
    +-------+-------------------+-------------------+-----------------+-----------------------------------------------------+
    | 0     | quantize          | 96x96x3 (float32) | 96x96x3 (int8)  | BuiltinOptionsType=0                                |
    | 1     | conv_2d           | 96x96x3 (int8)    | 48x48x8 (int8)  | Padding:same stride:2x2 activation:relu             |
    |       |                   | 3x3x3 (int8)      |                 |                                                     |
    |       |                   | 8 (int32)         |                 |                                                     |
    | 2     | depthwise_conv_2d | 48x48x8 (int8)    | 48x48x8 (int8)  | Multipler:1 padding:same stride:1x1 activation:relu |
    |       |                   | 3x3x8 (int8)      |                 |                                                     |
    |       |                   | 8 (int32)         |                 |                                                     |
    | 3     | conv_2d           | 48x48x8 (int8)    | 48x48x16 (int8) | Padding:same stride:1x1 activation:relu             |
    |       |                   | 1x1x8 (int8)      |                 |                                                     |
    |       |                   | 16 (int32)        |                 |                                                     |
    | 4     | depthwise_conv_2d | 48x48x16 (int8)   | 24x24x16 (int8) | Multipler:1 padding:same stride:2x2 activation:relu |
    |       |                   | 3x3x16 (int8)     |                 |                                                     |
    |       |                   | 16 (int32)        |                 |                                                     |
    | 5     | conv_2d           | 24x24x16 (int8)   | 24x24x32 (int8) | Padding:same stride:1x1 activation:relu             |
    |       |                   | 1x1x16 (int8)     |                 |                                                     |
    |       |                   | 32 (int32)        |                 |                                                     |
    | 6     | depthwise_conv_2d | 24x24x32 (int8)   | 24x24x32 (int8) | Multipler:1 padding:same stride:1x1 activation:relu |
    |       |                   | 3x3x32 (int8)     |                 |                                                     |
    |       |                   | 32 (int32)        |                 |                                                     |
    | 7     | conv_2d           | 24x24x32 (int8)   | 24x24x32 (int8) | Padding:same stride:1x1 activation:relu             |
    |       |                   | 1x1x32 (int8)     |                 |                                                     |
    |       |                   | 32 (int32)        |                 |                                                     |
    | 8     | depthwise_conv_2d | 24x24x32 (int8)   | 12x12x32 (int8) | Multipler:1 padding:same stride:2x2 activation:relu |
    |       |                   | 3x3x32 (int8)     |                 |                                                     |
    |       |                   | 32 (int32)        |                 |                                                     |
    | 9     | conv_2d           | 12x12x32 (int8)   | 12x12x64 (int8) | Padding:same stride:1x1 activation:relu             |
    |       |                   | 1x1x32 (int8)     |                 |                                                     |
    |       |                   | 64 (int32)        |                 |                                                     |
    | 10    | depthwise_conv_2d | 12x12x64 (int8)   | 12x12x64 (int8) | Multipler:1 padding:same stride:1x1 activation:relu |
    |       |                   | 3x3x64 (int8)     |                 |                                                     |
    |       |                   | 64 (int32)        |                 |                                                     |
    | 11    | conv_2d           | 12x12x64 (int8)   | 12x12x64 (int8) | Padding:same stride:1x1 activation:relu             |
    |       |                   | 1x1x64 (int8)     |                 |                                                     |
    |       |                   | 64 (int32)        |                 |                                                     |
    | 12    | depthwise_conv_2d | 12x12x64 (int8)   | 6x6x64 (int8)   | Multipler:1 padding:same stride:2x2 activation:relu |
    |       |                   | 3x3x64 (int8)     |                 |                                                     |
    |       |                   | 64 (int32)        |                 |                                                     |
    | 13    | conv_2d           | 6x6x64 (int8)     | 6x6x128 (int8)  | Padding:same stride:1x1 activation:relu             |
    |       |                   | 1x1x64 (int8)     |                 |                                                     |
    |       |                   | 128 (int32)       |                 |                                                     |
    | 14    | depthwise_conv_2d | 6x6x128 (int8)    | 6x6x128 (int8)  | Multipler:1 padding:same stride:1x1 activation:relu |
    |       |                   | 3x3x128 (int8)    |                 |                                                     |
    |       |                   | 128 (int32)       |                 |                                                     |
    | 15    | conv_2d           | 6x6x128 (int8)    | 6x6x128 (int8)  | Padding:same stride:1x1 activation:relu             |
    |       |                   | 1x1x128 (int8)    |                 |                                                     |
    |       |                   | 128 (int32)       |                 |                                                     |
    | 16    | depthwise_conv_2d | 6x6x128 (int8)    | 6x6x128 (int8)  | Multipler:1 padding:same stride:1x1 activation:relu |
    |       |                   | 3x3x128 (int8)    |                 |                                                     |
    |       |                   | 128 (int32)       |                 |                                                     |
    | 17    | conv_2d           | 6x6x128 (int8)    | 6x6x128 (int8)  | Padding:same stride:1x1 activation:relu             |
    |       |                   | 1x1x128 (int8)    |                 |                                                     |
    |       |                   | 128 (int32)       |                 |                                                     |
    | 18    | depthwise_conv_2d | 6x6x128 (int8)    | 6x6x128 (int8)  | Multipler:1 padding:same stride:1x1 activation:relu |
    |       |                   | 3x3x128 (int8)    |                 |                                                     |
    |       |                   | 128 (int32)       |                 |                                                     |
    | 19    | conv_2d           | 6x6x128 (int8)    | 6x6x128 (int8)  | Padding:same stride:1x1 activation:relu             |
    |       |                   | 1x1x128 (int8)    |                 |                                                     |
    |       |                   | 128 (int32)       |                 |                                                     |
    | 20    | depthwise_conv_2d | 6x6x128 (int8)    | 6x6x128 (int8)  | Multipler:1 padding:same stride:1x1 activation:relu |
    |       |                   | 3x3x128 (int8)    |                 |                                                     |
    |       |                   | 128 (int32)       |                 |                                                     |
    | 21    | conv_2d           | 6x6x128 (int8)    | 6x6x128 (int8)  | Padding:same stride:1x1 activation:relu             |
    |       |                   | 1x1x128 (int8)    |                 |                                                     |
    |       |                   | 128 (int32)       |                 |                                                     |
    | 22    | depthwise_conv_2d | 6x6x128 (int8)    | 6x6x128 (int8)  | Multipler:1 padding:same stride:1x1 activation:relu |
    |       |                   | 3x3x128 (int8)    |                 |                                                     |
    |       |                   | 128 (int32)       |                 |                                                     |
    | 23    | conv_2d           | 6x6x128 (int8)    | 6x6x128 (int8)  | Padding:same stride:1x1 activation:relu             |
    |       |                   | 1x1x128 (int8)    |                 |                                                     |
    |       |                   | 128 (int32)       |                 |                                                     |
    | 24    | depthwise_conv_2d | 6x6x128 (int8)    | 3x3x128 (int8)  | Multipler:1 padding:same stride:2x2 activation:relu |
    |       |                   | 3x3x128 (int8)    |                 |                                                     |
    |       |                   | 128 (int32)       |                 |                                                     |
    | 25    | conv_2d           | 3x3x128 (int8)    | 3x3x256 (int8)  | Padding:same stride:1x1 activation:relu             |
    |       |                   | 1x1x128 (int8)    |                 |                                                     |
    |       |                   | 256 (int32)       |                 |                                                     |
    | 26    | depthwise_conv_2d | 3x3x256 (int8)    | 3x3x256 (int8)  | Multipler:1 padding:same stride:1x1 activation:relu |
    |       |                   | 3x3x256 (int8)    |                 |                                                     |
    |       |                   | 256 (int32)       |                 |                                                     |
    | 27    | conv_2d           | 3x3x256 (int8)    | 3x3x256 (int8)  | Padding:same stride:1x1 activation:relu             |
    |       |                   | 1x1x256 (int8)    |                 |                                                     |
    |       |                   | 256 (int32)       |                 |                                                     |
    | 28    | average_pool_2d   | 3x3x256 (int8)    | 1x1x256 (int8)  | Padding:valid stride:3x3 filter:3x3 activation:none |
    | 29    | reshape           | 1x1x256 (int8)    | 256 (int8)      | BuiltinOptionsType=0                                |
    |       |                   | 2 (int32)         |                 |                                                     |
    | 30    | fully_connected   | 256 (int8)        | 2 (int8)        | Activation:none                                     |
    |       |                   | 256 (int8)        |                 |                                                     |
    |       |                   | 2 (int32)         |                 |                                                     |
    | 31    | softmax           | 2 (int8)          | 2 (int8)        | BuiltinOptionsType=9                                |
    | 32    | dequantize        | 2 (int8)          | 2 (float32)     | BuiltinOptionsType=0                                |
    +-------+-------------------+-------------------+-----------------+-----------------------------------------------------+
    Total MACs: 7.490 M
    Total OPs: 15.324 M
    Name: visual_wake_words
    Version: 1
    Description: TinyML: Visual Wake Words - MobileNetv1 with COCO14
    Classes: person, non_person
    hash: 0fdc40de5812cfa530f6ec120c55171a
    date: 2022-02-04T19:23:33.736Z
    runtime_memory_size: 156424
    samplewise_norm.rescale: 0.0
    samplewise_norm.mean_and_std: False
    .tflite file size: 334.2kB


Model Profiling Report
-----------------------

.. code-block:: shell
   
   # Profile on physical EFR32xG24 using MVP accelerator
   mltk profile visual_wake_words --device --accelerator MVP

    Profiling Summary
    Name: visual_wake_words
    Accelerator: MVP
    Input Shape: 1x96x96x3
    Input Data Type: float32
    Output Shape: 1x2
    Output Data Type: float32
    Flash, Model File Size (bytes): 334.1k
    RAM, Runtime Memory Size (bytes): 156.5k
    Operation Count: 15.8M
    Multiply-Accumulate Count: 7.5M
    Layer Count: 33
    Unsupported Layer Count: 0
    Accelerator Cycle Count: 7.5M
    CPU Cycle Count: 3.1M
    CPU Utilization (%): 32.4
    Clock Rate (hz): 78.0M
    Time (s): 123.2m
    Ops/s: 127.9M
    MACs/s: 60.8M
    Inference/s: 8.1

    Model Layers
    +-------+-------------------+--------+--------+------------+------------+----------+---------------------------+--------------+------------------------------------------------------+
    | Index | OpCode            | # Ops  | # MACs | Acc Cycles | CPU Cycles | Time (s) | Input Shape               | Output Shape | Options                                              |
    +-------+-------------------+--------+--------+------------+------------+----------+---------------------------+--------------+------------------------------------------------------+
    | 0     | quantize          | 110.6k | 0      | 0          | 941.1k     | 11.8m    | 1x96x96x3                 | 1x96x96x3    | Type=none                                            |
    | 1     | conv_2d           | 1.1M   | 497.7k | 695.3k     | 15.8k      | 8.8m     | 1x96x96x3,8x3x3x3,8       | 1x48x48x8    | Padding:same stride:2x2 activation:relu              |
    | 2     | depthwise_conv_2d | 387.1k | 165.9k | 583.3k     | 249.1k     | 9.4m     | 1x48x48x8,1x3x3x8,8       | 1x48x48x8    | Multiplier:1 padding:same stride:1x1 activation:relu |
    | 3     | conv_2d           | 700.4k | 294.9k | 341.1k     | 5.9k       | 4.3m     | 1x48x48x8,16x1x1x8,16     | 1x48x48x16   | Padding:same stride:1x1 activation:relu              |
    | 4     | depthwise_conv_2d | 193.5k | 82.9k  | 297.9k     | 257.2k     | 5.9m     | 1x48x48x16,1x3x3x16,16    | 1x24x24x16   | Multiplier:1 padding:same stride:2x2 activation:relu |
    | 5     | conv_2d           | 645.1k | 294.9k | 281.2k     | 5.9k       | 3.6m     | 1x24x24x16,32x1x1x16,32   | 1x24x24x32   | Padding:same stride:1x1 activation:relu              |
    | 6     | depthwise_conv_2d | 387.1k | 165.9k | 299.2k     | 806.6k     | 10.2m    | 1x24x24x32,1x3x3x32,32    | 1x24x24x32   | Multiplier:1 padding:same stride:1x1 activation:relu |
    | 7     | conv_2d           | 1.2M   | 589.8k | 507.0k     | 5.4k       | 6.4m     | 1x24x24x32,32x1x1x32,32   | 1x24x24x32   | Padding:same stride:1x1 activation:relu              |
    | 8     | depthwise_conv_2d | 96.8k  | 41.5k  | 74.8k      | 202.8k     | 2.6m     | 1x24x24x32,1x3x3x32,32    | 1x12x12x32   | Multiplier:1 padding:same stride:2x2 activation:relu |
    | 9     | conv_2d           | 617.5k | 294.9k | 251.2k     | 5.4k       | 3.2m     | 1x12x12x32,64x1x1x32,64   | 1x12x12x64   | Padding:same stride:1x1 activation:relu              |
    | 10    | depthwise_conv_2d | 193.5k | 82.9k  | 139.9k     | 207.9k     | 2.9m     | 1x12x12x64,1x3x3x64,64    | 1x12x12x64   | Multiplier:1 padding:same stride:1x1 activation:relu |
    | 11    | conv_2d           | 1.2M   | 589.8k | 474.7k     | 5.4k       | 6.0m     | 1x12x12x64,64x1x1x64,64   | 1x12x12x64   | Padding:same stride:1x1 activation:relu              |
    | 12    | depthwise_conv_2d | 48.4k  | 20.7k  | 35.0k      | 52.9k      | 750.0u   | 1x12x12x64,1x3x3x64,64    | 1x6x6x64     | Multiplier:1 padding:same stride:2x2 activation:relu |
    | 13    | conv_2d           | 603.6k | 294.9k | 236.3k     | 5.4k       | 3.0m     | 1x6x6x64,128x1x1x64,128   | 1x6x6x128    | Padding:same stride:1x1 activation:relu              |
    | 14    | depthwise_conv_2d | 96.8k  | 41.5k  | 62.2k      | 53.3k      | 1.1m     | 1x6x6x128,1x3x3x128,128   | 1x6x6x128    | Multiplier:1 padding:same stride:1x1 activation:relu |
    | 15    | conv_2d           | 1.2M   | 589.8k | 458.7k     | 5.4k       | 5.8m     | 1x6x6x128,128x1x1x128,128 | 1x6x6x128    | Padding:same stride:1x1 activation:relu              |
    | 16    | depthwise_conv_2d | 96.8k  | 41.5k  | 62.2k      | 53.3k      | 1.1m     | 1x6x6x128,1x3x3x128,128   | 1x6x6x128    | Multiplier:1 padding:same stride:1x1 activation:relu |
    | 17    | conv_2d           | 1.2M   | 589.8k | 458.7k     | 5.4k       | 5.8m     | 1x6x6x128,128x1x1x128,128 | 1x6x6x128    | Padding:same stride:1x1 activation:relu              |
    | 18    | depthwise_conv_2d | 96.8k  | 41.5k  | 62.2k      | 53.3k      | 1.1m     | 1x6x6x128,1x3x3x128,128   | 1x6x6x128    | Multiplier:1 padding:same stride:1x1 activation:relu |
    | 19    | conv_2d           | 1.2M   | 589.8k | 458.7k     | 5.4k       | 5.8m     | 1x6x6x128,128x1x1x128,128 | 1x6x6x128    | Padding:same stride:1x1 activation:relu              |
    | 20    | depthwise_conv_2d | 96.8k  | 41.5k  | 62.2k      | 53.3k      | 1.1m     | 1x6x6x128,1x3x3x128,128   | 1x6x6x128    | Multiplier:1 padding:same stride:1x1 activation:relu |
    | 21    | conv_2d           | 1.2M   | 589.8k | 458.7k     | 5.4k       | 5.8m     | 1x6x6x128,128x1x1x128,128 | 1x6x6x128    | Padding:same stride:1x1 activation:relu              |
    | 22    | depthwise_conv_2d | 96.8k  | 41.5k  | 62.2k      | 53.3k      | 1.1m     | 1x6x6x128,1x3x3x128,128   | 1x6x6x128    | Multiplier:1 padding:same stride:1x1 activation:relu |
    | 23    | conv_2d           | 1.2M   | 589.8k | 458.7k     | 5.4k       | 5.8m     | 1x6x6x128,128x1x1x128,128 | 1x6x6x128    | Padding:same stride:1x1 activation:relu              |
    | 24    | depthwise_conv_2d | 24.2k  | 10.4k  | 15.5k      | 14.2k      | 300.0u   | 1x6x6x128,1x3x3x128,128   | 1x3x3x128    | Multiplier:1 padding:same stride:2x2 activation:relu |
    | 25    | conv_2d           | 596.7k | 294.9k | 229.3k     | 5.4k       | 2.9m     | 1x3x3x128,256x1x1x128,256 | 1x3x3x256    | Padding:same stride:1x1 activation:relu              |
    | 26    | depthwise_conv_2d | 48.4k  | 20.7k  | 24.9k      | 14.2k      | 420.0u   | 1x3x3x256,1x3x3x256,256   | 1x3x3x256    | Multiplier:1 padding:same stride:1x1 activation:relu |
    | 27    | conv_2d           | 1.2M   | 589.8k | 451.3k     | 5.4k       | 5.7m     | 1x3x3x256,256x1x1x256,256 | 1x3x3x256    | Padding:same stride:1x1 activation:relu              |
    | 28    | average_pool_2d   | 2.6k   | 0      | 1.6k       | 3.9k       | 60.0u    | 1x3x3x256                 | 1x1x1x256    | Padding:valid stride:3x3 filter:3x3 activation:none  |
    | 29    | reshape           | 0      | 0      | 0          | 1.7k       | 0        | 1x1x1x256,2               | 1x256        | Type=none                                            |
    | 30    | fully_connected   | 1.0k   | 512.0  | 797.0      | 2.1k       | 60.0u    | 1x256,2x256,2             | 1x2          | Activation:none                                      |
    | 31    | softmax           | 10.0   | 0      | 0          | 2.8k       | 30.0u    | 1x2                       | 1x2          | Type=softmaxoptions                                  |
    | 32    | dequantize        | 4.0    | 0      | 0          | 938.0      | 0        | 1x2                       | 1x2          | Type=none                                            |
    +-------+-------------------+--------+--------+------------+------------+----------+---------------------------+--------------+------------------------------------------------------+


Model Diagram
------------------

.. code-block:: shell
   
   mltk view visual_wake_words --tflite

.. raw:: html

    <div class="model-diagram">
        <a href="../../../../_images/models/visual_wake_words.tflite.png" target="_blank">
            <img src="../../../../_images/models/visual_wake_words.tflite.png" />
            <p>Click to enlarge</p>
        </a>
    </div>

"""

from mltk.core.preprocess.image.parallel_generator import ParallelImageDataGenerator
from mltk.core.model import (
    MltkModel,
    TrainMixin,
    ImageDatasetMixin,
    EvaluateClassifierMixin
)
from mltk.models.shared import MobileNetV1
from mltk.utils.archive_downloader import download_verify_extract


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

#################################################
# General parameters
my_model.version = 1
my_model.description = 'TinyML: Visual Wake Words - MobileNetv1 with COCO14'

#################################################
# Training parameters
my_model.epochs = 50
my_model.batch_size = 50 
my_model.optimizer = 'adam'
my_model.metrics = ['accuracy']
my_model.loss = 'categorical_crossentropy'


#################################################
# Image Dataset Settings

# The directory of the training data

def download_data():
    return download_verify_extract(
        url='https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/vw_coco2014_96.tar.gz',
        dest_subdir='datasets/mscoco14/preprocessed/v1',
        file_hash='A5A465082D3F396407F8B5ABAF824DD5B28439C4',
        show_progress=True,
        remove_root_dir=True
    )


my_model.dataset = download_data
# The classification type
my_model.class_mode = 'categorical'
# The class labels found in your training dataset directory
my_model.classes =  ('person', 'non_person')
# The input shape to the model. The dataset samples will be resized if necessary
my_model.input_shape = (96, 96, 3)

validation_split = 0.1


##############################################################
# Training callbacks
#

# Learning rate schedule
def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 20:
        lrate = 0.0005
    if epoch > 30:
        lrate = 0.00025
    return lrate

my_model.lr_schedule = dict(
    schedule = lr_schedule,
    verbose = 1
)

my_model.datagen = ParallelImageDataGenerator(
    cores=.35,
    max_batches_pending=32,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=.1,
    horizontal_flip=True,
    validation_split=validation_split
)



##############################################################
# Model Layout
def my_model_builder(model: MyModel):
    keras_model = MobileNetV1(
        input_shape=model.input_shape
    )
    keras_model.compile(
        loss=model.loss, 
        optimizer=model.optimizer, 
        metrics=model.metrics
    )
    return keras_model

my_model.build_model_function = my_model_builder




##########################################################################################
# The following allows for running this model training script directly, e.g.: 
# python visual_wake_words.py
#
# Note that this has the same functionality as:
# mltk train visual_wake_words
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
    # This does the same as issuing the command: mltk train visual_wake_words-test --clean
    train_results = mltk_core.train_model(my_model, clean=True, test=test_mode_enabled)
    print(train_results)

    # Evaluate the model against the quantized .h5 (i.e. float32) model
    # This does the same as issuing the command: mltk evaluate visual_wake_words-test
    tflite_eval_results = mltk_core.evaluate_model(my_model, verbose=True, test=test_mode_enabled)
    print(tflite_eval_results)

    # Profile the model in the simulator
    # This does the same as issuing the command: mltk profile visual_wake_words-test
    profiling_results = mltk_core.profile_model(my_model, test=test_mode_enabled)
    print(profiling_results)