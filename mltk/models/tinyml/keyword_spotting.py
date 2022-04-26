"""keyword_spotting
**********************

Keyword spotting for 10 words

- Source code: `keyword_spotting.py <https://github.com/siliconlabs/mltk/blob/master/mltk/models/tinyml/keyword_spotting.py>`_
- Pre-trained model: `keyword_spotting.mltk.zip <https://github.com/siliconlabs/mltk/blob/master/mltk/models/tinyml/keyword_spotting.mltk.zip>`_


This was adapted from:

* https://github.com/mlcommons/tiny/tree/master/benchmark/training/keyword_spotting
* https://github.com/SiliconLabs/platform_ml_models/tree/master/eembc/KWS10_ARM_DSConv


Dataset
---------
* http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz

Model Topology
---------------
* https://arxiv.org/pdf/1711.07128.pdf
* https://github.com/ARM-software/ML-KWS-for-MCU

Spectrogram Characteristics
----------------------------
* Front-end: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/microfrontend
* Configuration: window=30ms, stride=20ms, bins=10, Upper frequency limit=4KHz

Performance (floating point model)
-----------------------------------
* Accuracy - 94.3%
* AUC -.998

Performance (quantized tflite model)
-------------------------------------
* Accuracy - 94.3%
* AUC - .997


Commands
--------------

.. code-block:: console

   # Do a "dry run" test training of the model
   > mltk train keyword_spotting-test

   # Train the model
   > mltk train keyword_spotting

   # Evaluate the trained model .tflite model
   > mltk evaluate keyword_spotting --tflite

   # Profile the model in the MVP hardware accelerator simulator
   > mltk profile keyword_spotting --accelerator MVP

   # Profile the model on a physical development board
   > mltk profile keyword_spotting --accelerator MVP --device


Model Summary
--------------

.. code-block:: console
    
    > mltk summarize keyword_spotting --tflite
    
    +-------+-------------------+-------------------+----------------+-------------------------------------------------------+  
    | Index | OpCode            | Input(s)          | Output(s)      | Config                                                |  
    +-------+-------------------+-------------------+----------------+-------------------------------------------------------+  
    | 0     | quantize          | 50x10x1 (float32) | 50x10x1 (int8) | BuiltinOptionsType=0                                  |  
    | 1     | conv_2d           | 50x10x1 (int8)    | 25x5x64 (int8) | Padding:same stride:2x2 activation:relu               |  
    |       |                   | 10x4x1 (int8)     |                |                                                       |  
    |       |                   | 64 (int32)        |                |                                                       |  
    | 2     | depthwise_conv_2d | 25x5x64 (int8)    | 25x5x64 (int8) | Multipler:1 padding:same stride:1x1 activation:relu   |  
    |       |                   | 3x3x64 (int8)     |                |                                                       |  
    |       |                   | 64 (int32)        |                |                                                       |  
    | 3     | conv_2d           | 25x5x64 (int8)    | 25x5x64 (int8) | Padding:same stride:1x1 activation:relu               |  
    |       |                   | 1x1x64 (int8)     |                |                                                       |  
    |       |                   | 64 (int32)        |                |                                                       |  
    | 4     | depthwise_conv_2d | 25x5x64 (int8)    | 25x5x64 (int8) | Multipler:1 padding:same stride:1x1 activation:relu   |  
    |       |                   | 3x3x64 (int8)     |                |                                                       |  
    |       |                   | 64 (int32)        |                |                                                       |  
    | 5     | conv_2d           | 25x5x64 (int8)    | 25x5x64 (int8) | Padding:same stride:1x1 activation:relu               |  
    |       |                   | 1x1x64 (int8)     |                |                                                       |  
    |       |                   | 64 (int32)        |                |                                                       |  
    | 6     | depthwise_conv_2d | 25x5x64 (int8)    | 25x5x64 (int8) | Multipler:1 padding:same stride:1x1 activation:relu   |  
    |       |                   | 3x3x64 (int8)     |                |                                                       |  
    |       |                   | 64 (int32)        |                |                                                       |  
    | 7     | conv_2d           | 25x5x64 (int8)    | 25x5x64 (int8) | Padding:same stride:1x1 activation:relu               |  
    |       |                   | 1x1x64 (int8)     |                |                                                       |  
    |       |                   | 64 (int32)        |                |                                                       |  
    | 8     | depthwise_conv_2d | 25x5x64 (int8)    | 25x5x64 (int8) | Multipler:1 padding:same stride:1x1 activation:relu   |  
    |       |                   | 3x3x64 (int8)     |                |                                                       |  
    |       |                   | 64 (int32)        |                |                                                       |  
    | 9     | conv_2d           | 25x5x64 (int8)    | 25x5x64 (int8) | Padding:same stride:1x1 activation:relu               |  
    |       |                   | 1x1x64 (int8)     |                |                                                       |  
    |       |                   | 64 (int32)        |                |                                                       |  
    | 10    | average_pool_2d   | 25x5x64 (int8)    | 1x1x64 (int8)  | Padding:valid stride:5x25 filter:5x25 activation:none |  
    | 11    | reshape           | 1x1x64 (int8)     | 64 (int8)      | BuiltinOptionsType=0                                  |  
    |       |                   | 2 (int32)         |                |                                                       |  
    | 12    | fully_connected   | 64 (int8)         | 12 (int8)      | Activation:none                                       |  
    |       |                   | 64 (int8)         |                |                                                       |  
    |       |                   | 12 (int32)        |                |                                                       |  
    | 13    | softmax           | 12 (int8)         | 12 (int8)      | BuiltinOptionsType=9                                  |  
    | 14    | dequantize        | 12 (int8)         | 12 (float32)   | BuiltinOptionsType=0                                  |  
    +-------+-------------------+-------------------+----------------+-------------------------------------------------------+  
    Total MACs: 2.657 M                                                                                                         
    Total OPs: 5.396 M                                                                                                          
    Name: keyword_spotting                                                                                                      
    Version: 1                                                                                                                  
    Description: TinyML: Keyword spotting for 10 words - dsconv_arm                                                             
    Classes: yes, no, up, down, left, right, on, off, stop, go, unknown, _background_noise_                                     
    hash: eb6e9d70cacfd495cdf36577882b83fc                                                                                      
    date: 2022-02-04T19:21:57.066Z                                                                                              
    runtime_memory_size: 21424                                                                                                  
    samplewise_norm.rescale: 0.0                                                                                                
    samplewise_norm.mean_and_std: False                                                                                         
    .tflite file size: 53.7kB


Model Diagram
------------------

.. code-block:: console
   
   > mltk view keyword_spotting --tflite

.. raw:: html

    <div class="model-diagram">
        <a href="../../../../_images/models/keyword_spotting.tflite.png" target="_blank">
            <img src="../../../../_images/models/keyword_spotting.tflite.png" />
            <p>Click to enlarge</p>
        </a>
    </div>

"""
# pylint: disable=redefined-outer-name



from mltk.core.preprocess.image.parallel_generator import ParallelImageDataGenerator
from mltk.core.model import (
    MltkModel,
    TrainMixin,
    ImageDatasetMixin,
    EvaluateClassifierMixin
)
from mltk.models.shared import DepthwiseSeparableConv2D_ARM
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



# General parameters
my_model.version = 1
my_model.description = 'TinyML: Keyword spotting for 10 words - dsconv_arm'


#################################################
# Training parameters
my_model.epochs = 80
my_model.batch_size = 64 
my_model.optimizer = 'adam'
my_model.metrics = ['accuracy']
my_model.loss = 'categorical_crossentropy'


#################################################
# Image Dataset Settings

# The directory of the training data



def download_dataset():
    """Load the dataset
    """

    path = download_verify_extract(
        url='https://github.com/SiliconLabs/mltk_assets/raw/master/datasets/speech_dataset_spec.7z',
        dest_subdir='datasets/speech_commands/preprocessed/v1',
        file_hash='20E36646073492FDB4FB8285EC49042E70F9E60E',
        show_progress=True,
        remove_root_dir=True
    )
    return path

my_model.dataset = download_dataset
# The classification type
my_model.class_mode = 'categorical'
# The class labels found in your training dataset directory
my_model.classes =  ('yes','no','up','down','left','right','on','off','stop','go','unknown','_background_noise_')
# The input shape to the model. The dataset samples will be resized if necessary
my_model.input_shape =(50, 10, 1)

validation_split = 0.1


##############################################################
# Training callbacks
#

# Learning rate schedule
def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 30:
        lrate = 0.0005
    if epoch > 40:
        lrate = 0.00025
    if epoch > 50:
        lrate = 0.00025
    if epoch > 60:
        lrate = 0.0001
    return lrate

my_model.lr_schedule = dict(
    schedule = lr_schedule,
    verbose = 1
)


my_model.datagen = ParallelImageDataGenerator(
    cores=.35,
    max_batches_pending=32,
    rotation_range=0,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=(0.95, 1.05),
    #brightness_range=(0.9, 1.2),
    #contrast_range=(0.9, 1.2),
    validation_split=validation_split
)



##############################################################
# Model Layout
def my_model_builder(model: MyModel):
    keras_model = DepthwiseSeparableConv2D_ARM(
        input_shape=model.input_shape
    )
    keras_model.compile(
        loss=model.loss, 
        optimizer=model.optimizer, 
        metrics=model.metrics
    )
    return keras_model

my_model.build_model_function  = my_model_builder