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

.. code-block:: console

   # Do a "dry run" test training of the model
   > mltk train vww_model-test

   # Train the model
   > mltk train vww_model

   # Evaluate the trained model .tflite model
   > mltk evaluate vww_model --tflite

   # Profile the model in the MVP hardware accelerator simulator
   > mltk profile vww_model --accelerator MVP

   # Profile the model on a physical development board
   > mltk profile vww_model --accelerator MVP --device


Model Summary
--------------

.. code-block:: console
    
    > mltk summarize visual_wake_words --tflite
    
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


Model Diagram
------------------

.. code-block:: console
   
   > mltk view visual_wake_words --tflite

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