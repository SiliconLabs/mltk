"""audio_example1
********************

- Source code: `audio_example1.py <https://github.com/siliconlabs/mltk/blob/master/mltk/models/examples/audio_example1.py>`_
- Pre-trained model archive: `audio_example1.mltk.zip <https://github.com/siliconlabs/mltk/blob/master/mltk/models/examples/audio_example1.mltk.zip>`_

This provides an example of how to create a Keyword Search (KWS) classification model.
This example defines a model to detect the keywords:

* left
* right
* up 
* down

This use the Google speech_commands dataset with the :py:class:`mltk.core.preprocess.audio.parallel_generator.ParallelAudioDataGenerator`


Commands
--------------

.. code-block:: shell

   # Do a "dry run" test training of the model
   mltk train audio_example1-test

   # Train the model
   mltk train audio_example1

   # Evaluate the trained model .tflite model
   mltk evaluate audio_example1 --tflite

   # Profile the model in the MVP hardware accelerator simulator
   mltk profile audio_example1 --accelerator MVP

   # Profile the model on a physical development board
   mltk profile audio_example1 --accelerator MVP --device

   # Run this model in the audio classifier application
   mltk classify_audio audio_example1 --device --verbose

Model Summary
--------------

.. code-block:: shell
    
    mltk summarize audio_example1 --tflite
    
    +-------+-------------------+-----------------+-----------------+-----------------------------------------------------+
    | Index | OpCode            | Input(s)        | Output(s)       | Config                                              |
    +-------+-------------------+-----------------+-----------------+-----------------------------------------------------+
    | 0     | depthwise_conv_2d | 59x49x1 (int8)  | 30x25x8 (int8)  | Multipler:8 padding:same stride:2x2 activation:relu |
    |       |                   | 7x7x8 (int8)    |                 |                                                     |
    |       |                   | 8 (int32)       |                 |                                                     |
    | 1     | conv_2d           | 30x25x8 (int8)  | 14x12x24 (int8) | Padding:valid stride:2x2 activation:relu            |
    |       |                   | 3x3x8 (int8)    |                 |                                                     |
    |       |                   | 24 (int32)      |                 |                                                     |
    | 2     | max_pool_2d       | 14x12x24 (int8) | 7x6x24 (int8)   | Padding:valid stride:2x2 filter:2x2 activation:none |
    | 3     | conv_2d           | 7x6x24 (int8)   | 5x4x20 (int8)   | Padding:valid stride:1x1 activation:relu            |
    |       |                   | 3x3x24 (int8)   |                 |                                                     |
    |       |                   | 20 (int32)      |                 |                                                     |
    | 4     | max_pool_2d       | 5x4x20 (int8)   | 2x2x20 (int8)   | Padding:valid stride:2x2 filter:2x2 activation:none |
    | 5     | reshape           | 2x2x20 (int8)   | 80 (int8)       | BuiltinOptionsType=0                                |
    |       |                   | 2 (int32)       |                 |                                                     |
    | 6     | fully_connected   | 80 (int8)       | 6 (int8)        | Activation:none                                     |
    |       |                   | 80 (int8)       |                 |                                                     |
    |       |                   | 6 (int32)       |                 |                                                     |
    | 7     | softmax           | 6 (int8)        | 6 (int8)        | BuiltinOptionsType=9                                |
    +-------+-------------------+-----------------+-----------------+-----------------------------------------------------+
    Total MACs: 671.184 k
    Total OPs: 1.357 M
    Name: audio_example1
    Version: 1
    Description: Audio classifier example for detecting left/right/up/down keywords
    Classes: up, down, left, right, _unknown_, _silence_
    hash: b8d28debb3af3495c6d8d2a67eedfa22
    date: 2022-02-03T22:56:08.361Z
    runtime_memory_size: 12052
    average_window_duration_ms: 1000
    detection_threshold: 165
    suppression_ms: 1500
    minimum_count: 3
    volume_db: 5.0
    latency_ms: 0
    log_level: info
    samplewise_norm.rescale: 0.0
    samplewise_norm.mean_and_std: False
    fe.sample_rate_hz: 16000
    fe.sample_length_ms: 1200
    fe.window_size_ms: 30
    fe.window_step_ms: 20
    fe.filterbank_n_channels: 49
    fe.filterbank_upper_band_limit: 3999.0
    fe.filterbank_lower_band_limit: 125.0
    fe.noise_reduction_enable: True
    fe.noise_reduction_smoothing_bits: 10
    fe.noise_reduction_even_smoothing: 0.02500000037252903
    fe.noise_reduction_odd_smoothing: 0.05999999865889549
    fe.noise_reduction_min_signal_remaining: 0.029999999329447746
    fe.pcan_enable: False
    fe.pcan_strength: 0.949999988079071
    fe.pcan_offset: 80.0
    fe.pcan_gain_bits: 21
    fe.log_scale_enable: True
    fe.log_scale_shift: 6
    fe.fft_length: 512
    .tflite file size: 16.0kB


Model Profiling Report
-----------------------

.. code-block:: shell
   
   # Profile on physical EFR32xG24 using MVP accelerator
   mltk profile audio_example1 --device --accelerator MVP

    Profiling Summary
    Name: audio_example1
    Accelerator: MVP
    Input Shape: 1x59x49x1
    Input Data Type: int8
    Output Shape: 1x6
    Output Data Type: int8
    Flash, Model File Size (bytes): 16.0k
    RAM, Runtime Memory Size (bytes): 12.3k
    Operation Count: 1.4M
    Multiply-Accumulate Count: 671.2k
    Layer Count: 8
    Unsupported Layer Count: 0
    Accelerator Cycle Count: 1.0M
    CPU Cycle Count: 350.9k
    CPU Utilization (%): 28.4
    Clock Rate (hz): 80.0M
    Time (s): 15.4m
    Ops/s: 89.2M
    MACs/s: 43.4M
    Inference/s: 64.7

    Model Layers
    +-------+-------------------+--------+--------+------------+------------+----------+-----------------------+--------------+------------------------------------------------------+
    | Index | OpCode            | # Ops  | # MACs | Acc Cycles | CPU Cycles | Time (s) | Input Shape           | Output Shape | Options                                              |
    +-------+-------------------+--------+--------+------------+------------+----------+-----------------------+--------------+------------------------------------------------------+
    | 0     | depthwise_conv_2d | 606.0k | 294.0k | 434.5k     | 287.5k     | 7.5m     | 1x59x49x1,1x7x7x8,8   | 1x30x25x8    | Multiplier:8 padding:same stride:2x2 activation:relu |
    | 1     | conv_2d           | 592.7k | 290.3k | 459.7k     | 6.6k       | 5.7m     | 1x30x25x8,24x3x3x8,24 | 1x14x12x24   | Padding:valid stride:2x2 activation:relu             |
    | 2     | max_pool_2d       | 4.0k   | 0      | 3.2k       | 21.6k      | 270.0u   | 1x14x12x24            | 1x7x6x24     | Padding:valid stride:2x2 filter:2x2 activation:none  |
    | 3     | conv_2d           | 174.0k | 86.4k  | 132.2k     | 8.6k       | 1.7m     | 1x7x6x24,20x3x3x24,20 | 1x5x4x20     | Padding:valid stride:1x1 activation:relu             |
    | 4     | max_pool_2d       | 320.0  | 0      | 380.0      | 18.0k      | 210.0u   | 1x5x4x20              | 1x2x2x20     | Padding:valid stride:2x2 filter:2x2 activation:none  |
    | 5     | reshape           | 0      | 0      | 0          | 878.0      | 30.0u    | 1x2x2x20,2            | 1x80         | Type=none                                            |
    | 6     | fully_connected   | 966.0  | 480.0  | 776.0      | 2.1k       | 30.0u    | 1x80,6x80,6           | 1x6          | Activation:none                                      |
    | 7     | softmax           | 30.0   | 0      | 0          | 5.7k       | 60.0u    | 1x6                   | 1x6          | Type=softmaxoptions                                  |
    +-------+-------------------+--------+--------+------------+------------+----------+-----------------------+--------------+------------------------------------------------------+


Model Diagram
------------------

.. code-block:: shell
   
   mltk view audio_example1 --tflite

.. raw:: html

    <div class="model-diagram">
        <a href="../../../../_images/models/audio_example1.tflite.png" target="_blank">
            <img src="../../../../_images/models/audio_example1.tflite.png" />
            <p>Click to enlarge</p>
        </a>
    </div>


"""
# pylint: disable=redefined-outer-name
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, 
    Activation, 
    Flatten, 
    BatchNormalization,
    Conv2D,
    DepthwiseConv2D,
    MaxPooling2D
)


from mltk.core import (
    MltkModel,
    TrainMixin,
    AudioDatasetMixin,
    EvaluateClassifierMixin,
    TrainingResults
)
from mltk.core.preprocess.audio.parallel_generator import ParallelAudioDataGenerator
from mltk.core.preprocess.audio.audio_feature_generator import AudioFeatureGeneratorSettings
from mltk.datasets.audio.speech_commands import speech_commands_v2


# Instantiate the model object with the following 'mixins':
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
my_model = MyModel()


#################################################
# General Settings
my_model.version = 1
my_model.description = 'Audio classifier example for detecting left/right/up/down keywords'

#################################################
# Training Basic Settings
my_model.epochs = -1 # We use the EarlyStopping keras callback to stop the training
my_model.batch_size = 64 
my_model.optimizer = 'adam'
my_model.metrics = ['accuracy']
my_model.loss = 'categorical_crossentropy'


#################################################
# Training callback Settings

my_model.tensorboard['write_images'] = True 

my_model.checkpoint['monitor'] =  'val_accuracy'

# https://keras.io/api/callbacks/reduce_lr_on_plateau/
# If the test accuracy doesn't improve after 'patience' epochs 
# then decrease the learning rate by 'factor'
my_model.reduce_lr_on_plateau = dict(
  monitor='accuracy',
  factor = 0.25,
  patience = 4 # NOTE: In practice, this should be larger (e.g 15) but it will increase training times
)

# https://keras.io/api/callbacks/early_stopping/
# If the validation accuracy doesn't improve after 'patience' epochs then stop training
my_model.early_stopping = dict( 
  monitor = 'val_accuracy',
  patience = 15 # NOTE: In practice, this should be larger (e.g. 45) but it will increase training times
)


#################################################
# TF-Lite converter settings
my_model.tflite_converter['optimizations'] = [tf.lite.Optimize.DEFAULT]
my_model.tflite_converter['supported_ops'] = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
my_model.tflite_converter['inference_input_type'] = np.int8
my_model.tflite_converter['inference_output_type'] = np.int8
 # generate a representative dataset from the validation data
my_model.tflite_converter['representative_dataset'] = 'generate'



#################################################
# Audio Data Provider Settings
my_model.dataset = speech_commands_v2
my_model.class_mode = 'categorical'
my_model.classes = ['up', 'down', 'left', 'right', '_unknown_', '_silence_']

# The numbers of samples for each class is different
# Then ensures each class contributes equally to training the model
my_model.class_weights = 'balanced'



#################################################
# AudioFeatureGenerator Settings
# See https://siliconlabs.github.io/mltk/docs/audio/audio_feature_generator.html
frontend_settings = AudioFeatureGeneratorSettings()
frontend_settings.sample_rate_hz = 16000
frontend_settings.sample_length_ms = 1200
frontend_settings.window_size_ms = 30
frontend_settings.window_step_ms = 20
frontend_settings.filterbank_n_channels = 49
frontend_settings.filterbank_upper_band_limit = 4000-1
frontend_settings.filterbank_lower_band_limit = 125.0
frontend_settings.noise_reduction_enable = True
frontend_settings.noise_reduction_smoothing_bits = 10
frontend_settings.noise_reduction_even_smoothing = 0.025
frontend_settings.noise_reduction_odd_smoothing = 0.06
frontend_settings.noise_reduction_min_signal_remaining = 0.03
frontend_settings.pcan_enable = False
frontend_settings.pcan_strength = 0.95
frontend_settings.pcan_offset = 80.0
frontend_settings.pcan_gain_bits = 21
frontend_settings.log_scale_enable = True
frontend_settings.log_scale_shift = 6


#################################################
# ParallelAudioDataGenerator Settings

my_model.datagen = ParallelAudioDataGenerator(
    dtype=my_model.tflite_converter['inference_input_type'],
    frontend_settings=frontend_settings,
    cores=0.45,
    debug=False, # Set this to true to enable debugging of the generator
    max_batches_pending=32, 
    validation_split= 0.15,
    validation_augmentation_enabled=False,
    samplewise_center=False,
    samplewise_std_normalization=False,
    rescale=None,
    unknown_class_percentage=2, # In practice, this should be larger (e.g. 4.0) but it will increase training times
    silence_class_percentage=0.2,
    offset_range=(0.0,1.0),
    trim_threshold_db=20,
    noise_colors=None,
#     loudness_range=(0.6, 3.0), # In practice, these should be enabled but they will increase training times 
#     speed_range=(0.6,1.9),
#     pitch_range=(0.7,1.4),
#     vtlp_range=(0.7,1.4),
    bg_noise_range=(0.0,0.3),
    bg_noise_dir='_background_noise_'
)


#################################################
# Model Layout
def my_model_builder(model: MyModel):
    keras_model = Sequential(name=model.name)
    
    keras_model.add(DepthwiseConv2D(kernel_size=(7,7), 
                            depth_multiplier=8, 
                            strides=(2,2), 
                            use_bias=True,
                            padding='same',
                            input_shape=model.input_shape))
    keras_model.add(BatchNormalization())
    keras_model.add(Activation('relu'))
    
    keras_model.add(Conv2D(kernel_size=(3,3), 
                            filters=24, 
                            strides=(2,2), 
                            use_bias=True,
                            padding='valid'))
    keras_model.add(BatchNormalization())
    keras_model.add(Activation('relu'))

    keras_model.add(MaxPooling2D(pool_size=(2,2), 
                                  strides=(2,2,), 
                                  padding='valid'))

    keras_model.add(Conv2D(kernel_size=(3,3), 
                            filters=20, 
                            strides=(1,1), 
                            use_bias=True,
                            padding='valid'))
    keras_model.add(BatchNormalization())
    keras_model.add(Activation('relu'))

    keras_model.add(MaxPooling2D(pool_size=(2,2), 
                                  strides=(2,2,), 
                                  padding='valid'))

    keras_model.add(Flatten())
    keras_model.add(Dense(model.n_classes, activation='softmax'))
    keras_model.compile(loss=model.loss, 
                         optimizer=model.optimizer, 
                         metrics=model.metrics)

    return keras_model

my_model.build_model_function = my_model_builder



def _on_training_complete(results: TrainingResults):
    """This callback is invoked after training successfully completes
    
    Here is where custom quantization or .tflite model generation could be done
    """
    name, score = results.get_best_metric()

    tflite_path = results.mltk_model.tflite_archive_path
    h5_path = results.mltk_model.h5_archive_path

    print('\n')
    print('*' * 60)
    print('_on_training_complete callback:\n')
    print(f'Best metric: {name}: {score}')
    print(f'saved .tflite path: {tflite_path}')
    print(f'saved .h5 path: {h5_path}')
    print('*' * 60)
    print('\n')


my_model.on_training_complete = _on_training_complete


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
my_model.model_parameters['detection_threshold'] = 165

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