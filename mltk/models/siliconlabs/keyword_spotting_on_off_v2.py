"""keyword_spotting_on_off_v2
*******************************

- Source code: `keyword_spotting_on_off_v2.py <https://github.com/siliconlabs/mltk/blob/master/mltk/models/siliconlabs/keyword_spotting_on_off_v2.py>`_
- Pre-trained model: `keyword_spotting_on_off_v2.mltk.zip <https://github.com/siliconlabs/mltk/blob/master/mltk/models/siliconlabs/keyword_spotting_on_off_v2.mltk.zip>`_


This model specification script is designed to work with the
`Keyword Spotting On/Off <https://siliconlabs.github.io/mltk/mltk/tutorials/keyword_spotting_on_off.html>`_ tutorial.

This model is a CNN classifier to detect the keywords:

- on
- off

Changes from v1 
----------------
This model is based on `keyword_spotting_on_off.py <https://github.com/siliconlabs/mltk/blob/master/mltk/models/siliconlabs/keyword_spotting_on_off.py>`_
but has the following changes:

- Model architecture uses more convolutional filters: This gives the model better accuracy at the expense of RAM and execution latency
- Audio Feature Generator "activity detection" block used: Inference only runs if activity is detected in the audio stream
- Dynamic spectrogram scaling: The audio spectrogram is quantized from uint16 to int8 using a dynamic scaling method
- DC Notch Filter: A DC notch filter is applied tot he audio signal

Dataset
---------
This uses the :py:class:`mltk.datasets.audio.speech_commands.speech_commands_v2` dataset provided by Google.

Preprocessing
--------------
This uses the :py:class:`mltk.core.preprocess.audio.parallel_generator.ParallelAudioDataGenerator` with the
:py:class:`mltk.core.preprocess.audio.audio_feature_generator.AudioFeatureGenerator` settings:

- sample_rate: 8kHz
- sample_length: 1.0s
- window size: 32ms
- window step: 16ms
- n_channels: 64


Commands
--------------

.. code-block:: shell

   # Do a "dry run" test training of the model
   mltk train keyword_spotting_on_off_v2-test

   # Train the model
   mltk train keyword_spotting_on_off_v2

   # Evaluate the trained model .tflite model
   mltk evaluate keyword_spotting_on_off_v2 --tflite

   # Profile the model in the MVP hardware accelerator simulator
   mltk profile keyword_spotting_on_off_v2 --accelerator MVP

   # Profile the model on a physical development board
   mltk profile keyword_spotting_on_off_v2  --accelerator MVP --device

   # Run the model in the audio classifier on the local PC
   mltk classify_audio keyword_spotting_on_off_v2 --verbose

   # Run the model in the audio classifier on the physical device
   mltk classify_audio keyword_spotting_on_off_v2 --device --verbose


Model Summary
--------------

.. code-block:: shell
    
    mltk summarize keyword_spotting_on_off --tflite
    
    +-------+-----------------+-----------------+-----------------+-----------------------------------------------------+
    | Index | OpCode          | Input(s)        | Output(s)       | Config                                              |
    +-------+-----------------+-----------------+-----------------+-----------------------------------------------------+
    | 0     | conv_2d         | 61x64x1 (int8)  | 61x64x8 (int8)  | Padding:same stride:1x1 activation:relu             |
    |       |                 | 3x3x1 (int8)    |                 |                                                     |
    |       |                 | 8 (int32)       |                 |                                                     |
    | 1     | max_pool_2d     | 61x64x8 (int8)  | 30x32x8 (int8)  | Padding:valid stride:2x2 filter:2x2 activation:none |
    | 2     | conv_2d         | 30x32x8 (int8)  | 30x32x16 (int8) | Padding:same stride:1x1 activation:relu             |
    |       |                 | 3x3x8 (int8)    |                 |                                                     |
    |       |                 | 16 (int32)      |                 |                                                     |
    | 3     | max_pool_2d     | 30x32x16 (int8) | 15x16x16 (int8) | Padding:valid stride:2x2 filter:2x2 activation:none |
    | 4     | conv_2d         | 15x16x16 (int8) | 15x16x32 (int8) | Padding:same stride:1x1 activation:relu             |
    |       |                 | 3x3x16 (int8)   |                 |                                                     |
    |       |                 | 32 (int32)      |                 |                                                     |
    | 5     | max_pool_2d     | 15x16x32 (int8) | 7x8x32 (int8)   | Padding:valid stride:2x2 filter:2x2 activation:none |
    | 6     | conv_2d         | 7x8x32 (int8)   | 7x8x32 (int8)   | Padding:same stride:1x1 activation:relu             |
    |       |                 | 3x3x32 (int8)   |                 |                                                     |
    |       |                 | 32 (int32)      |                 |                                                     |
    | 7     | max_pool_2d     | 7x8x32 (int8)   | 3x4x32 (int8)   | Padding:valid stride:2x2 filter:2x2 activation:none |
    | 8     | conv_2d         | 3x4x32 (int8)   | 3x4x32 (int8)   | Padding:same stride:1x1 activation:relu             |
    |       |                 | 3x3x32 (int8)   |                 |                                                     |
    |       |                 | 32 (int32)      |                 |                                                     |
    | 9     | max_pool_2d     | 3x4x32 (int8)   | 1x2x32 (int8)   | Padding:valid stride:2x2 filter:2x2 activation:none |
    | 10    | reshape         | 1x2x32 (int8)   | 64 (int8)       | Type=none                                           |
    |       |                 | 2 (int32)       |                 |                                                     |
    | 11    | fully_connected | 64 (int8)       | 3 (int8)        | Activation:none                                     |
    |       |                 | 64 (int8)       |                 |                                                     |
    |       |                 | 3 (int32)       |                 |                                                     |
    | 12    | softmax         | 3 (int8)        | 3 (int8)        | Type=softmaxoptions                                 |
    +-------+-----------------+-----------------+-----------------+-----------------------------------------------------+
    Total MACs: 3.120 M
    Total OPs: 6.351 M
    Name: keyword_spotting_on_off_v2
    Version: 2
    Description: Keyword spotting classifier to detect: "on" and "off"
    Classes: on, off, _unknown_
    hash: fc9279e225e30f02617f944515c73381
    date: 2022-05-19T20:06:40.343Z
    runtime_memory_size: 42460
    average_window_duration_ms: 1000
    detection_threshold: 160
    suppression_ms: 100
    minimum_count: 0
    volume_gain: 0.0
    latency_ms: 100
    verbose_model_output_logs: False
    samplewise_norm.rescale: 0.0
    samplewise_norm.mean_and_std: False
    fe.sample_rate_hz: 8000
    fe.sample_length_ms: 1000
    fe.window_size_ms: 32
    fe.window_step_ms: 16
    fe.filterbank_n_channels: 64
    fe.filterbank_upper_band_limit: 3999.0
    fe.filterbank_lower_band_limit: 100.0
    fe.noise_reduction_enable: False
    fe.noise_reduction_smoothing_bits: 5
    fe.noise_reduction_even_smoothing: 0.004000000189989805
    fe.noise_reduction_odd_smoothing: 0.004000000189989805
    fe.noise_reduction_min_signal_remaining: 0.05000000074505806
    fe.pcan_enable: False
    fe.pcan_strength: 0.949999988079071
    fe.pcan_offset: 80.0
    fe.pcan_gain_bits: 21
    fe.log_scale_enable: True
    fe.log_scale_shift: 6
    fe.activity_detection_enable: True
    fe.activity_detection_alpha_a: 0.5
    fe.activity_detection_alpha_b: 0.800000011920929
    fe.activity_detection_arm_threshold: 0.75
    fe.activity_detection_trip_threshold: 0.800000011920929
    fe.dc_notch_filter_enable: True
    fe.dc_notch_filter_coefficient: 0.949999988079071
    fe.quantize_dynamic_scale_enable: True
    fe.quantize_dynamic_scale_range_db: 40.0
    fe.fft_length: 256
    .tflite file size: 39.6kB


Model Profiling Report
-----------------------

.. code-block:: shell
   
   # Profile on physical EFR32xG24 using MVP accelerator
   mltk profile keyword_spotting_on_off_v2 --device --accelerator MVP

    Profiling Summary
    Name: keyword_spotting_on_off_v2
    Accelerator: MVP
    Input Shape: 1x61x64x1
    Input Data Type: int8
    Output Shape: 1x3
    Output Data Type: int8
    Flash, Model File Size (bytes): 39.6k
    RAM, Runtime Memory Size (bytes): 42.5k
    Operation Count: 6.4M
    Multiply-Accumulate Count: 3.1M
    Layer Count: 13
    Unsupported Layer Count: 0
    Accelerator Cycle Count: 5.0M
    CPU Cycle Count: 254.3k
    CPU Utilization (%): 5.0
    Clock Rate (hz): 80.0M
    Time (s): 63.5m
    Ops/s: 101.5M
    MACs/s: 48.9M
    Inference/s: 15.7

    Model Layers
    +-------+-----------------+--------+--------+------------+------------+----------+-------------------------+--------------+-----------------------------------------------------+
    | Index | OpCode          | # Ops  | # MACs | Acc Cycles | CPU Cycles | Time (s) | Input Shape             | Output Shape | Options                                             |
    +-------+-----------------+--------+--------+------------+------------+----------+-------------------------+--------------+-----------------------------------------------------+
    | 0     | conv_2d         | 655.9k | 281.1k | 981.7k     | 28.8k      | 12.2m    | 1x61x64x1,8x3x3x1,8     | 1x61x64x8    | Padding:same stride:1x1 activation:relu             |
    | 1     | max_pool_2d     | 30.7k  | 0      | 23.1k      | 8.2k       | 330.0u   | 1x61x64x8               | 1x30x32x8    | Padding:valid stride:2x2 filter:2x2 activation:none |
    | 2     | conv_2d         | 2.2M   | 1.1M   | 1.7M       | 28.8k      | 20.8m    | 1x30x32x8,16x3x3x8,16   | 1x30x32x16   | Padding:same stride:1x1 activation:relu             |
    | 3     | max_pool_2d     | 15.4k  | 0      | 11.6k      | 15.0k      | 210.0u   | 1x30x32x16              | 1x15x16x16   | Padding:valid stride:2x2 filter:2x2 activation:none |
    | 4     | conv_2d         | 2.2M   | 1.1M   | 1.6M       | 29.2k      | 19.3m    | 1x15x16x16,32x3x3x16,32 | 1x15x16x32   | Padding:same stride:1x1 activation:relu             |
    | 5     | max_pool_2d     | 7.2k   | 0      | 5.6k       | 28.2k      | 360.0u   | 1x15x16x32              | 1x7x8x32     | Padding:valid stride:2x2 filter:2x2 activation:none |
    | 6     | conv_2d         | 1.0M   | 516.1k | 652.8k     | 29.2k      | 8.1m     | 1x7x8x32,32x3x3x32,32   | 1x7x8x32     | Padding:same stride:1x1 activation:relu             |
    | 7     | max_pool_2d     | 1.5k   | 0      | 1.4k       | 28.1k      | 330.0u   | 1x7x8x32                | 1x3x4x32     | Padding:valid stride:2x2 filter:2x2 activation:none |
    | 8     | conv_2d         | 222.3k | 110.6k | 109.8k     | 24.1k      | 1.5m     | 1x3x4x32,32x3x3x32,32   | 1x3x4x32     | Padding:same stride:1x1 activation:relu             |
    | 9     | max_pool_2d     | 256.0  | 0      | 416.0      | 28.1k      | 330.0u   | 1x3x4x32                | 1x1x2x32     | Padding:valid stride:2x2 filter:2x2 activation:none |
    | 10    | reshape         | 0      | 0      | 0          | 799.0      | 30.0u    | 1x1x2x32,2              | 1x64         | Type=none                                           |
    | 11    | fully_connected | 387.0  | 192.0  | 328.0      | 2.1k       | 30.0u    | 1x64,3x64,3             | 1x3          | Activation:none                                     |
    | 12    | softmax         | 15.0   | 0      | 0          | 3.7k       | 30.0u    | 1x3                     | 1x3          | Type=softmaxoptions                                 |
    +-------+-----------------+--------+--------+------------+------------+----------+-------------------------+--------------+-----------------------------------------------------+


Model Diagram
------------------

.. code-block:: shell
   
   mltk view keyword_spotting_on_off_v2 --tflite

.. raw:: html

    <div class="model-diagram">
        <a href="../../../../_images/models/keyword_spotting_on_off_v2.tflite.png" target="_blank">
            <img src="../../../../_images/models/keyword_spotting_on_off_v2.tflite.png" />
            <p>Click to enlarge</p>
        </a>
    </div>



License
---------
This model was developed by Silicon Labs and is covered by a standard 
`Silicon Labs MSLA <https://www.silabs.com/about-us/legal/master-software-license-agreement>`_.

"""
# pylint: disable=redefined-outer-name

# Import the Tensorflow packages
# required to build the model layout
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, 
    Activation, 
    Flatten, 
    BatchNormalization,
    Conv2D,
    MaxPooling2D,
    Dropout
)


# Import the MLTK model object 
# and necessary mixins
# Later in this script we configure the various properties
from mltk.core import (
    MltkModel,
    TrainMixin,
    AudioDatasetMixin,
    EvaluateClassifierMixin
)

# Import the Google speech_commands dataset package
# This manages downloading and extracting the dataset
from mltk.datasets.audio.speech_commands import speech_commands_v2

# Import the ParallelAudioDataGenerator
# This has two main jobs:
# 1. Process the Google speech_commands dataset and apply random augmentations during training
# 2. Generate a spectrogram using the AudioFeatureGenerator from each augmented audio sample 
#    and give the spectrogram to Tensorflow for model training
from mltk.core.preprocess.audio.parallel_generator import ParallelAudioDataGenerator
# Import the AudioFeatureGeneratorSettings which we'll configure 
# and give to the ParallelAudioDataGenerator
from mltk.core.preprocess.audio.audio_feature_generator import AudioFeatureGeneratorSettings



# Define a custom model object with the following 'mixins':
# - TrainMixin        - Provides classifier model training operations and settings
# - AudioDatasetMixin - Provides audio data generation operations and settings
# - EvaluateClassifierMixin     - Provides classifier evaluation operations and settings
# @mltk_model # NOTE: This tag is required for this model be discoverable
class MyModel(
    MltkModel, 
    TrainMixin, 
    AudioDatasetMixin, 
    EvaluateClassifierMixin
):
    pass

# Instantiate our custom model object
# The rest of this script simply configures the properties
# of our custom model object
my_model = MyModel()


#################################################
# General Settings

# For better tracking, the version should be incremented any time a non-trivial change is made
# NOTE: The version is optional and not used directly used by the MLTK
my_model.version = 2
# Provide a brief description about what this model models
# This description goes in the "description" field of the .tflite model file
my_model.description = 'Keyword spotting classifier to detect: "on" and "off"'

#################################################
# Training Basic Settings

# This specifies the number of times we run the training
# samples through the model to update the model weights.
# Typically, a larger value leads to better accuracy at the expense of training time.
# Set to -1 to use the early_stopping callback and let the scripts
# determine how many epochs to train for (see below).
# Otherwise set this to a specific value (typically 40-200)
my_model.epochs = 80
# Specify how many samples to pass through the model
# before updating the training gradients.
# Typical values are 10-64
# NOTE: Larger values require more memory and may not fit on your GPU
my_model.batch_size = 10 
# This specifies the algorithm used to update the model gradients
# during training. Adam is very common
# See https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
my_model.optimizer = 'adam' 
# List of metrics to be evaluated by the model during training and testing
my_model.metrics = ['accuracy']
# The "loss" function used to update the weights
# This is a classification problem with more than two labels so we use categorical_crossentropy
# See https://www.tensorflow.org/api_docs/python/tf/keras/losses
my_model.loss = 'categorical_crossentropy'


#################################################
# Training callback Settings

# Generate checkpoints every time the validation accuracy improves
# See https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
my_model.checkpoint['monitor'] =  'val_accuracy'

# If the training accuracy doesn't improve after 'patience' epochs 
# then decrease the learning rate by 'factor'
# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau
# NOTE: Alternatively, we could define our own learn rate schedule
#       using my_model.lr_schedule
# my_model.reduce_lr_on_plateau = dict(
#  monitor='accuracy',
#  factor = 0.25,
#  patience = 10
#)

# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler
# Update the learning rate each epoch based on the given callback
def lr_schedule(epoch):
    initial_learning_rate = 0.001
    decay_per_epoch = 0.95
    lrate = initial_learning_rate * (decay_per_epoch ** epoch)
    return lrate

my_model.lr_schedule = dict(
    schedule = lr_schedule,
    verbose = 1
)


# If the validation accuracy doesn't improve after 'patience' epochs 
# then stop training, the epochs must be -1 to use this callback
# See https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
# my_model.early_stopping['monitor'] = 'val_accuracy'
# my_model.early_stopping['patience'] = 12 # Increasing this may improve accuracy at the expense of training time


#################################################
# TF-Lite converter settings

# These are the settings used to quantize the model
# We want all the internal ops as well as
# model input/output to be int8
my_model.tflite_converter['optimizations'] = [tf.lite.Optimize.DEFAULT]
my_model.tflite_converter['supported_ops'] = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# NOTE: A float32 model input/output is also possible
my_model.tflite_converter['inference_input_type'] = np.int8 
my_model.tflite_converter['inference_output_type'] = np.int8
# Automatically generate a representative dataset from the validation data
my_model.tflite_converter['representative_dataset'] = 'generate'



#################################################
# Audio Data Provider Settings

# Specify the dataset 
# NOTE: This can also be an absolute path to a directory
#       or a Python function
# See: https://siliconlabs.github.io/mltk/docs/python_api/core/mltk_model.html#mltk.core.AudioDatasetMixin.dataset
my_model.dataset = speech_commands_v2
# We're using a 'categorical_crossentropy' loss
# so must also use a `categorical` class mode for the data generation
my_model.class_mode = 'categorical'

# Specify the keywords we want to detect
# In this model, we detect "on" and "off",
# plus two pseudo classes: _unknown_ and _silence_
#
# Any number of classes may be added here as long as they're
# found in the dataset specified above.
# NOTE: You'll likely need a larger model for more classes
my_model.classes = ['on', 'off', '_unknown_']

# The numbers of samples for each class is different
# Then ensures each class contributes equally to training the model
my_model.class_weights = 'balanced'


#################################################
# AudioFeatureGenerator Settings
# 
# These are the settings used by the AudioFeatureGenerator 
# to generate spectrograms from the audio samples
# These settings must be used during modeling training
# AND by embedded device at runtime
#
# See https://siliconlabs.github.io/mltk/docs/audio/audio_feature_generator.html
frontend_settings = AudioFeatureGeneratorSettings()

frontend_settings.sample_rate_hz = 8000  # This can also be 16k for slightly better performance at the cost of more RAM
frontend_settings.sample_length_ms = 1000
frontend_settings.window_size_ms = 32
frontend_settings.window_step_ms = 16
frontend_settings.filterbank_n_channels = 64
frontend_settings.filterbank_upper_band_limit = 4000.0-1 # Spoken language usually only goes up to 4k
frontend_settings.filterbank_lower_band_limit = 100.0
frontend_settings.noise_reduction_enable = False # Disable the noise reduction block
frontend_settings.noise_reduction_smoothing_bits = 5
frontend_settings.noise_reduction_even_smoothing = 0.004
frontend_settings.noise_reduction_odd_smoothing = 0.004
frontend_settings.noise_reduction_min_signal_remaining = 0.05
frontend_settings.pcan_enable = False
frontend_settings.pcan_strength = 0.95
frontend_settings.pcan_offset = 80.0
frontend_settings.pcan_gain_bits = 21
frontend_settings.log_scale_enable = True
frontend_settings.log_scale_shift = 6


frontend_settings.activity_detection_enable = True # Enable the activity detection block
frontend_settings.activity_detection_alpha_a = 0.5
frontend_settings.activity_detection_alpha_b = 0.8
frontend_settings.activity_detection_arm_threshold = 0.75
frontend_settings.activity_detection_trip_threshold = 0.8

frontend_settings.dc_notch_filter_enable = True # Enable the DC notch filter
frontend_settings.dc_notch_filter_coefficient = 0.95

frontend_settings.quantize_dynamic_scale_enable = True # Enable dynamic quantization
frontend_settings.quantize_dynamic_scale_range_db = 40.0


#################################################
# ParallelAudioDataGenerator Settings
#
# Configure the data generator settings
# This specifies how to augment the training samples
# See the command: "mltk view_audio"
# to get a better idea of how these augmentations affect
# the samples
my_model.datagen = ParallelAudioDataGenerator(
    dtype=my_model.tflite_converter['inference_input_type'],
    frontend_settings=frontend_settings,
    cores=0.45, # Adjust this as necessary for your PC setup
    debug=False, # Set this to true to enable debugging of the generator
    max_batches_pending=16,  # Adjust this as necessary for your PC setup (smaller -> less RAM)
    validation_split= 0.10,
    validation_augmentation_enabled=True,
    samplewise_center=False,
    samplewise_std_normalization=False,
    rescale=None,
    unknown_class_percentage=2.0, # Increasing this may help model robustness at the expense of training time
   # silence_class_percentage=0.3,
    offset_range=(0.0,1.0),
    trim_threshold_db=200,
    noise_colors=None,
    loudness_range=(0.2, 1.0),
    speed_range=(0.9,1.1),
    pitch_range=(-3,3),
    #vtlp_range=(0.9,1.1),
    bg_noise_range=(0.1,0.4),
    bg_noise_dir='_background_noise_' # This is a directory provided by the google speech commands dataset, can also provide an absolute path
)




#################################################
# Model Layout
#
# This defines the actual model layout 
# using the Keras API.
# This particular model is a relatively standard
# sequential Convolution Neural Network (CNN).
#
# It is important to the note the usage of the 
# "model" argument.
# Rather than hardcode values, the model is
# used to build the model, e.g.:
# Dense(model.n_classes)
#
# This way, the various model properties above can be modified
# without having to re-write this section.
#
def my_model_builder(model: MyModel):
    weight_decay = 1e-4
    regularizer = regularizers.l2(weight_decay)
    input_shape = model.input_shape
    filters = 8 
 
    keras_model = Sequential(name=model.name, layers = [
        Conv2D(filters, (3,3), padding='same', input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(2,2),

        Conv2D(2*filters,(3,3), padding='same'), 
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(2,2),

        Conv2D(4*filters, (3,3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(2,2),
    
        Conv2D(4*filters, (3,3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(2,2),

        Conv2D(4*filters, (3,3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(2,2),

        Dropout(0.5),

        Flatten(),
        Dense(model.n_classes, activation='softmax')
    ])

    keras_model.compile(
        loss=model.loss, 
        optimizer=model.optimizer, 
        metrics=model.metrics
    )

    return keras_model

my_model.build_model_function = my_model_builder




#################################################
# Audio Classifier Settings
#
# These are additional parameters to include in
# the generated .tflite model file.
# The settings are used by the "classify_audio" command
# or audio_classifier example application.
# NOTE: Corresponding command-line options will override these values.


# Controls the smoothing. 
# Drop all inference results that are older than <now> minus window_duration
# Longer durations (in milliseconds) will give a higher confidence that the results are correct, but may miss some commands
my_model.model_parameters['average_window_duration_ms'] = 1000

# Minimum averaged model output threshold for a class to be considered detected, 0-255. Higher values increase precision at the cost of recall
my_model.model_parameters['detection_threshold'] = 160

# Amount of milliseconds to wait after a keyword is detected before detecting new keywords
# Since we're using the audio detection block, we want this to be as short as possible
my_model.model_parameters['suppression_ms'] = 1

# The minimum number of inference results to average when calculating the detection value
my_model.model_parameters['minimum_count'] = 0 # Set to zero to disable averaging and just use the latest result

# Set the volume gain scaler (i.e. amplitude) to apply to the microphone data. If 0 or omitted, no scaler is applied
my_model.model_parameters['volume_gain'] = 0.0

# This the amount of time in milliseconds between audio processing loops
# Since we're using the audio detection block, we want this to be as short as possible
my_model.model_parameters['latency_ms'] = 1

# Enable verbose inference results
my_model.model_parameters['verbose_model_output_logs'] = False