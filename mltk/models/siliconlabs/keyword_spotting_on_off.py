"""keyword_spotting_on_off
**************************

- Source code: `keyword_spotting_on_off.py <https://github.com/siliconlabs/mltk/blob/master/mltk/models/siliconlabs/keyword_spotting_on_off.py>`_
- Pre-trained model: `keyword_spotting_on_off.mltk.zip <https://github.com/siliconlabs/mltk/blob/master/mltk/models/siliconlabs/keyword_spotting_on_off.mltk.zip>`_


This model specification script is designed to work with the
`Keyword Spotting On/Off <https://siliconlabs.github.io/mltk/mltk/tutorials/keyword_spotting_on_off.html>`_ tutorial.

This model is a CNN classifier to detect the keywords:

- on
- off

Dataset
---------
This uses the :py:class:`mltk.datasets.audio.speech_commands.speech_commands_v2` dataset provided by Google.

Preprocessing
--------------
This uses the :py:class:`mltk.core.preprocess.audio.parallel_generator.ParallelAudioDataGenerator` with the
:py:class:`mltk.core.preprocess.audio.audio_feature_generator.AudioFeatureGenerator` settings:

- sample_rate: 8kHz
- sample_length: 1.0s
- window size: 30ms
- window step: 20ms
- n_channels: 32


Commands
--------------

.. code-block:: shell

   # Do a "dry run" test training of the model
   mltk train keyword_spotting_on_off-test

   # Train the model
   mltk train keyword_spotting_on_off

   # Evaluate the trained model .tflite model
   mltk evaluate keyword_spotting_on_off --tflite

   # Profile the model in the MVP hardware accelerator simulator
   mltk profile keyword_spotting_on_off --accelerator MVP

   # Profile the model on a physical development board
   mltk profile keyword_spotting_on_off  --accelerator MVP --device

   # Run the model in the audio classifier on the local PC
   mltk classify_audio keyword_spotting_on_off --verbose

   # Run the model in the audio classifier on the physical device
   mltk classify_audio keyword_spotting_on_off --device --verbose


Model Summary
--------------

.. code-block:: shell
    
    mltk summarize keyword_spotting_on_off --tflite
    
    +-------+-----------------+----------------+----------------+-----------------------------------------------------+
    | Index | OpCode          | Input(s)       | Output(s)      | Config                                              |
    +-------+-----------------+----------------+----------------+-----------------------------------------------------+
    | 0     | conv_2d         | 49x32x1 (int8) | 25x16x8 (int8) | Padding:same stride:2x2 activation:relu             |
    |       |                 | 3x3x1 (int8)   |                |                                                     |
    |       |                 | 8 (int32)      |                |                                                     |
    | 1     | conv_2d         | 25x16x8 (int8) | 13x8x16 (int8) | Padding:same stride:2x2 activation:relu             |
    |       |                 | 3x3x8 (int8)   |                |                                                     |
    |       |                 | 16 (int32)     |                |                                                     |
    | 2     | conv_2d         | 13x8x16 (int8) | 7x4x32 (int8)  | Padding:same stride:2x2 activation:relu             |
    |       |                 | 3x3x16 (int8)  |                |                                                     |
    |       |                 | 32 (int32)     |                |                                                     |
    | 3     | max_pool_2d     | 7x4x32 (int8)  | 1x4x32 (int8)  | Padding:valid stride:1x7 filter:1x7 activation:none |
    | 4     | reshape         | 1x4x32 (int8)  | 128 (int8)     | BuiltinOptionsType=0                                |
    |       |                 | 2 (int32)      |                |                                                     |
    | 5     | fully_connected | 128 (int8)     | 4 (int8)       | Activation:none                                     |
    |       |                 | 128 (int8)     |                |                                                     |
    |       |                 | 4 (int32)      |                |                                                     |
    | 6     | softmax         | 4 (int8)       | 4 (int8)       | BuiltinOptionsType=9                                |
    +-------+-----------------+----------------+----------------+-----------------------------------------------------+
    Total MACs: 278.144 k
    Total OPs: 563.084 k
    Name: keyword_spotting_on_off
    Version: 1
    Description: Keyword spotting classifier to detect: "on" and "off"
    Classes: on, off, _unknown_, _silence_
    hash: 782baa4c65acec0db85a71d2be78eb29
    date: 2022-02-04T19:05:11.747Z
    runtime_memory_size: 6712
    average_window_duration_ms: 1000
    detection_threshold: 160
    suppression_ms: 750
    minimum_count: 3
    volume_db: 5.0
    latency_ms: 0
    log_level: info
    samplewise_norm.rescale: 0.0
    samplewise_norm.mean_and_std: False
    fe.sample_rate_hz: 8000
    fe.sample_length_ms: 1000
    fe.window_size_ms: 30
    fe.window_step_ms: 20
    fe.filterbank_n_channels: 32
    fe.filterbank_upper_band_limit: 3999.0
    fe.filterbank_lower_band_limit: 100.0
    fe.noise_reduction_enable: True
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
    fe.fft_length: 256
    .tflite file size: 15.3kB


Model Profiling Report
-----------------------

.. code-block:: shell
   
   # Profile on physical EFR32xG24 using MVP accelerator
   mltk profile keyword_spotting_on_off --device --accelerator MVP

    Profiling Summary
    Name: keyword_spotting_on_off
    Accelerator: MVP
    Input Shape: 1x49x32x1
    Input Data Type: int8
    Output Shape: 1x4
    Output Data Type: int8
    Flash, Model File Size (bytes): 15.3k
    RAM, Runtime Memory Size (bytes): 13.4k
    Operation Count: 574.5k
    Multiply-Accumulate Count: 278.1k
    Layer Count: 7
    Unsupported Layer Count: 0
    Accelerator Cycle Count: 224.3k
    CPU Cycle Count: 98.9k
    CPU Utilization (%): 34.6
    Clock Rate (hz): 78.0M
    Time (s): 3.7m
    Ops/s: 157.0M
    MACs/s: 76.0M
    Inference/s: 273.2

    Model Layers
    +-------+-----------------+--------+--------+------------+------------+----------+------------------------+--------------+-----------------------------------------------------+
    | Index | OpCode          | # Ops  | # MACs | Acc Cycles | CPU Cycles | Time (s) | Input Shape            | Output Shape | Options                                             |
    +-------+-----------------+--------+--------+------------+------------+----------+------------------------+--------------+-----------------------------------------------------+
    | 0     | conv_2d         | 67.2k  | 28.8k  | 46.7k      | 20.6k      | 720.0u   | 1x49x32x1,8x3x3x1,8    | 1x25x16x8    | Padding:same stride:2x2 activation:relu             |
    | 1     | conv_2d         | 244.6k | 119.8k | 90.8k      | 20.7k      | 1.3m     | 1x25x16x8,16x3x3x8,16  | 1x13x8x16    | Padding:same stride:2x2 activation:relu             |
    | 2     | conv_2d         | 260.7k | 129.0k | 85.2k      | 20.2k      | 1.2m     | 1x13x8x16,32x3x3x16,32 | 1x7x4x32     | Padding:same stride:2x2 activation:relu             |
    | 3     | max_pool_2d     | 896.0  | 0      | 800.0      | 30.0k      | 390.0u   | 1x7x4x32               | 1x1x4x32     | Padding:valid stride:1x7 filter:1x7 activation:none |
    | 4     | reshape         | 0      | 0      | 0          | 1.1k       | 30.0u    | 1x1x4x32,2             | 1x128        | Type=none                                           |
    | 5     | fully_connected | 1.0k   | 512.0  | 809.0      | 2.2k       | 30.0u    | 1x128,4x128,4          | 1x4          | Activation:none                                     |
    | 6     | softmax         | 20.0   | 0      | 0          | 4.1k       | 60.0u    | 1x4                    | 1x4          | Type=softmaxoptions                                 |
    +-------+-----------------+--------+--------+------------+------------+----------+------------------------+--------------+-----------------------------------------------------+


Model Diagram
------------------

.. code-block:: shell
   
   mltk view keyword_spotting_on_off --tflite

.. raw:: html

    <div class="model-diagram">
        <a href="../../../../_images/models/keyword_spotting_on_off.tflite.png" target="_blank">
            <img src="../../../../_images/models/keyword_spotting_on_off.tflite.png" />
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
my_model.version = 1 
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
# See: https://siliconlabs.github.io/mltk/docs/python_api/mltk_model/audio_dataset_mixin.html#mltk.core.AudioDatasetMixin.dataset
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
my_model.classes = ['on', 'off', '_unknown_', '_silence_']

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
frontend_settings.window_size_ms = 30
frontend_settings.window_step_ms = 20
frontend_settings.filterbank_n_channels = 32
frontend_settings.filterbank_upper_band_limit = 4000.0-1 # Spoken language usually only goes up to 4k
frontend_settings.filterbank_lower_band_limit = 100.0
frontend_settings.noise_reduction_enable = True
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
    silence_class_percentage=0.3,
    offset_range=(0.0,1.0),
    trim_threshold_db=30,
    noise_colors=None,
    loudness_range=(0.2, 1.0),
    speed_range=(0.9,1.1),
    pitch_range=(0.9,1.1),
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
        Conv2D(filters, (3,3), 
            padding='same', 
            kernel_regularizer=regularizer, 
            input_shape=input_shape, 
            strides=(2,2)
        ),
        BatchNormalization(),
        Activation('relu'),

        Conv2D(2*filters, (3,3), 
            padding='same', 
            kernel_regularizer=regularizer, 
            strides=(2,2)
        ),
        BatchNormalization(),
        Activation('relu'),
        Dropout(rate=0.1),

        Conv2D(4*filters, (3,3), 
            padding='same', 
            kernel_regularizer=regularizer, 
            strides=(2,2)
        ),
        BatchNormalization(),
        Activation('relu'),
        Dropout(rate=0.3),
        
        MaxPooling2D(pool_size=[7,1]),
        
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
my_model.model_parameters['suppression_ms'] = 750

# The minimum number of inference results to average when calculating the detection value
my_model.model_parameters['minimum_count'] = 3

# Set the volume gain scaler (i.e. amplitude) to apply to the microphone data. If 0 or omitted, no scaler is applied
my_model.model_parameters['volume_gain'] = 2

# This the amount of time in milliseconds an audio loop takes.
my_model.model_parameters['latency_ms'] = 100

# Enable verbose inference results
my_model.model_parameters['verbose_model_output_logs'] = False



##########################################################################################
# The following allows for running this model training script directly, e.g.: 
# python keyword_spotting_on_off.py
#
# Note that this has the same functionality as:
# mltk train keyword_spotting_on_off
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
    # This does the same as issuing the command: mltk train keyword_spotting_on_off-test --clean
    train_results = mltk_core.train_model(my_model, clean=True, test=test_mode_enabled)
    print(train_results)

    # Evaluate the model against the quantized .h5 (i.e. float32) model
    # This does the same as issuing the command: mltk evaluate keyword_spotting_on_off-test
    tflite_eval_results = mltk_core.evaluate_model(my_model, verbose=True, test=test_mode_enabled)
    print(tflite_eval_results)

    # Profile the model in the simulator
    # This does the same as issuing the command: mltk profile keyword_spotting_on_off-test
    profiling_results = mltk_core.profile_model(my_model, test=test_mode_enabled)
    print(profiling_results)