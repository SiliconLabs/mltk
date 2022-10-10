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

.. code-block:: shell

   # Do a "dry run" test training of the model
   mltk train keyword_spotting-test

   # Train the model
   mltk train keyword_spotting

   # Evaluate the trained model .tflite model
   mltk evaluate keyword_spotting --tflite

   # Profile the model in the MVP hardware accelerator simulator
   mltk profile keyword_spotting --accelerator MVP

   # Profile the model on a physical development board
   mltk profile keyword_spotting --accelerator MVP --device


Model Summary
--------------

.. code-block:: shell
    
    mltk summarize keyword_spotting --tflite
    
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


Model Profiling Report
-----------------------

.. code-block:: shell
   
   # Profile on physical EFR32xG24 using MVP accelerator
   mltk profile keyword_spotting --device --accelerator MVP

    Profiling Summary
    Name: keyword_spotting
    Accelerator: MVP
    Input Shape: 1x50x10x1
    Input Data Type: float32
    Output Shape: 1x12
    Output Data Type: float32
    Flash, Model File Size (bytes): 53.7k
    RAM, Runtime Memory Size (bytes): 37.6k
    Operation Count: 5.5M
    Multiply-Accumulate Count: 2.7M
    Layer Count: 15
    Unsupported Layer Count: 0
    Accelerator Cycle Count: 2.5M
    CPU Cycle Count: 831.4k
    CPU Utilization (%): 29.2
    Clock Rate (hz): 78.0M
    Time (s): 36.5m
    Ops/s: 150.7M
    MACs/s: 72.7M
    Inference/s: 27.4

    Model Layers
    +-------+-------------------+--------+--------+------------+------------+----------+------------------------+--------------+-------------------------------------------------------+
    | Index | OpCode            | # Ops  | # MACs | Acc Cycles | CPU Cycles | Time (s) | Input Shape            | Output Shape | Options                                               |
    +-------+-------------------+--------+--------+------------+------------+----------+------------------------+--------------+-------------------------------------------------------+
    | 0     | quantize          | 2.0k   | 0      | 0          | 17.9k      | 210.0u   | 1x50x10x1              | 1x50x10x1    | Type=none                                             |
    | 1     | conv_2d           | 664.0k | 320.0k | 382.0k     | 50.1k      | 5.0m     | 1x50x10x1,64x10x4x1,64 | 1x25x5x64    | Padding:same stride:2x2 activation:relu               |
    | 2     | depthwise_conv_2d | 168.0k | 72.0k  | 116.0k     | 181.1k     | 2.5m     | 1x25x5x64,1x3x3x64,64  | 1x25x5x64    | Multiplier:1 padding:same stride:1x1 activation:relu  |
    | 3     | conv_2d           | 1.0M   | 512.0k | 412.1k     | 5.5k       | 5.2m     | 1x25x5x64,64x1x1x64,64 | 1x25x5x64    | Padding:same stride:1x1 activation:relu               |
    | 4     | depthwise_conv_2d | 168.0k | 72.0k  | 116.0k     | 180.8k     | 2.5m     | 1x25x5x64,1x3x3x64,64  | 1x25x5x64    | Multiplier:1 padding:same stride:1x1 activation:relu  |
    | 5     | conv_2d           | 1.0M   | 512.0k | 412.1k     | 5.4k       | 5.2m     | 1x25x5x64,64x1x1x64,64 | 1x25x5x64    | Padding:same stride:1x1 activation:relu               |
    | 6     | depthwise_conv_2d | 168.0k | 72.0k  | 116.0k     | 180.8k     | 2.5m     | 1x25x5x64,1x3x3x64,64  | 1x25x5x64    | Multiplier:1 padding:same stride:1x1 activation:relu  |
    | 7     | conv_2d           | 1.0M   | 512.0k | 412.1k     | 5.4k       | 5.2m     | 1x25x5x64,64x1x1x64,64 | 1x25x5x64    | Padding:same stride:1x1 activation:relu               |
    | 8     | depthwise_conv_2d | 168.0k | 72.0k  | 116.0k     | 180.8k     | 2.5m     | 1x25x5x64,1x3x3x64,64  | 1x25x5x64    | Multiplier:1 padding:same stride:1x1 activation:relu  |
    | 9     | conv_2d           | 1.0M   | 512.0k | 412.1k     | 5.4k       | 5.2m     | 1x25x5x64,64x1x1x64,64 | 1x25x5x64    | Padding:same stride:1x1 activation:relu               |
    | 10    | average_pool_2d   | 8.1k   | 0      | 4.1k       | 3.9k       | 90.0u    | 1x25x5x64              | 1x1x1x64     | Padding:valid stride:5x25 filter:5x25 activation:none |
    | 11    | reshape           | 0      | 0      | 0          | 779.0      | 30.0u    | 1x1x1x64,2             | 1x64         | Type=none                                             |
    | 12    | fully_connected   | 1.5k   | 768.0  | 1.2k       | 2.1k       | 30.0u    | 1x64,12x64,12          | 1x12         | Activation:none                                       |
    | 13    | softmax           | 60.0   | 0      | 0          | 9.7k       | 120.0u   | 1x12                   | 1x12         | Type=softmaxoptions                                   |
    | 14    | dequantize        | 24.0   | 0      | 0          | 1.6k       | 30.0u    | 1x12                   | 1x12         | Type=none                                             |
    +-------+-------------------+--------+--------+------------+------------+----------+------------------------+--------------+-------------------------------------------------------+


Model Diagram
------------------

.. code-block:: shell
   
   mltk view keyword_spotting --tflite

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



##########################################################################################
# The following allows for running this model training script directly, e.g.: 
# python keyword_spotting.py
#
# Note that this has the same functionality as:
# mltk train keyword_spotting
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
    # This does the same as issuing the command: mltk train keyword_spotting-test --clean
    train_results = mltk_core.train_model(my_model, clean=True, test=test_mode_enabled)
    print(train_results)

    # Evaluate the model against the quantized .h5 (i.e. float32) model
    # This does the same as issuing the command: mltk evaluate keyword_spotting-test
    tflite_eval_results = mltk_core.evaluate_model(my_model, verbose=True, test=test_mode_enabled)
    print(tflite_eval_results)

    # Profile the model in the simulator
    # This does the same as issuing the command: mltk profile keyword_spotting-test
    profiling_results = mltk_core.profile_model(my_model, test=test_mode_enabled)
    print(profiling_results)