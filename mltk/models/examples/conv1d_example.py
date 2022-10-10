"""conv1d_example
********************

Source code: `conv1d_example.py <https://github.com/siliconlabs/mltk/blob/master/mltk/models/examples/conv1d_example.py>`_

This demonstrates how to create a Conv1D model.

This uses the ParallelAudioDataGenerator with the AudioFeatureGenerator disabled.


Commands
--------------

.. code-block:: shell

   # Do a "dry run" test training of the model
   mltk train conv1d_example-test

   # Train the model
   mltk train conv1d_example

   # Evaluate the trained model .tflite model
   mltk evaluate conv1d_example --tflite

   # Profile the model in the MVP hardware accelerator simulator
   mltk profile conv1d_example --accelerator MVP

   # Profile the model on a physical development board
   mltk profile conv1d_example --accelerator MVP --device

   # Directly invoke the model script
   python conv1d_example.py


Model Summary
--------------

.. code-block:: shell
    
    mltk summarize conv1d_example --tflite --build
    
    +-------+-----------------+------------------+------------------+-----------------------------------------------------+
    | Index | OpCode          | Input(s)         | Output(s)        | Config                                              |
    +-------+-----------------+------------------+------------------+-----------------------------------------------------+
    | 0     | expand_dims     | 3600x1 (int8)    | 1x3600x1 (int8)  | BuiltinOptionsType=52                               |
    |       |                 |  (int32)         |                  |                                                     |
    | 1     | conv_2d         | 1x3600x1 (int8)  | 1x900x16 (int8)  | Padding:same stride:4x1 activation:relu             |
    |       |                 | 1x7x1 (int8)     |                  |                                                     |
    |       |                 | 16 (int32)       |                  |                                                     |
    | 2     | reshape         | 1x900x16 (int8)  | 900x16 (int8)    | BuiltinOptionsType=0                                |
    |       |                 | 3 (int32)        |                  |                                                     |
    | 3     | expand_dims     | 900x16 (int8)    | 900x1x16 (int8)  | BuiltinOptionsType=52                               |
    |       |                 |  (int32)         |                  |                                                     |
    | 4     | max_pool_2d     | 900x1x16 (int8)  | 450x1x16 (int8)  | Padding:valid stride:1x2 filter:1x2 activation:none |
    | 5     | reshape         | 450x1x16 (int8)  | 450x16 (int8)    | BuiltinOptionsType=0                                |
    |       |                 | 3 (int32)        |                  |                                                     |
    | 6     | expand_dims     | 450x16 (int8)    | 1x450x16 (int8)  | BuiltinOptionsType=52                               |
    |       |                 |  (int32)         |                  |                                                     |
    | 7     | conv_2d         | 1x450x16 (int8)  | 1x450x32 (int8)  | Padding:same stride:1x1 activation:relu             |
    |       |                 | 1x5x16 (int8)    |                  |                                                     |
    |       |                 | 32 (int32)       |                  |                                                     |
    | 8     | reshape         | 1x450x32 (int8)  | 450x32 (int8)    | BuiltinOptionsType=0                                |
    |       |                 | 3 (int32)        |                  |                                                     |
    | 9     | expand_dims     | 450x32 (int8)    | 450x1x32 (int8)  | BuiltinOptionsType=52                               |
    |       |                 |  (int32)         |                  |                                                     |
    | 10    | max_pool_2d     | 450x1x32 (int8)  | 225x1x32 (int8)  | Padding:valid stride:1x2 filter:1x2 activation:none |
    | 11    | reshape         | 225x1x32 (int8)  | 225x32 (int8)    | BuiltinOptionsType=0                                |
    |       |                 | 3 (int32)        |                  |                                                     |
    | 12    | expand_dims     | 225x32 (int8)    | 1x225x32 (int8)  | BuiltinOptionsType=52                               |
    |       |                 |  (int32)         |                  |                                                     |
    | 13    | conv_2d         | 1x225x32 (int8)  | 1x225x64 (int8)  | Padding:same stride:1x1 activation:relu             |
    |       |                 | 1x3x32 (int8)    |                  |                                                     |
    |       |                 | 64 (int32)       |                  |                                                     |
    | 14    | reshape         | 1x225x64 (int8)  | 225x64 (int8)    | BuiltinOptionsType=0                                |
    |       |                 | 3 (int32)        |                  |                                                     |
    | 15    | expand_dims     | 225x64 (int8)    | 225x1x64 (int8)  | BuiltinOptionsType=52                               |
    |       |                 |  (int32)         |                  |                                                     |
    | 16    | max_pool_2d     | 225x1x64 (int8)  | 112x1x64 (int8)  | Padding:valid stride:1x2 filter:1x2 activation:none |
    | 17    | reshape         | 112x1x64 (int8)  | 112x64 (int8)    | BuiltinOptionsType=0                                |
    |       |                 | 3 (int32)        |                  |                                                     |
    | 18    | expand_dims     | 112x64 (int8)    | 1x112x64 (int8)  | BuiltinOptionsType=52                               |
    |       |                 |  (int32)         |                  |                                                     |
    | 19    | conv_2d         | 1x112x64 (int8)  | 1x112x128 (int8) | Padding:same stride:1x1 activation:relu             |
    |       |                 | 1x3x64 (int8)    |                  |                                                     |
    |       |                 | 128 (int32)      |                  |                                                     |
    | 20    | reshape         | 1x112x128 (int8) | 112x128 (int8)   | BuiltinOptionsType=0                                |
    |       |                 | 3 (int32)        |                  |                                                     |
    | 21    | expand_dims     | 112x128 (int8)   | 112x1x128 (int8) | BuiltinOptionsType=52                               |
    |       |                 |  (int32)         |                  |                                                     |
    | 22    | max_pool_2d     | 112x1x128 (int8) | 56x1x128 (int8)  | Padding:valid stride:1x2 filter:1x2 activation:none |
    | 23    | reshape         | 56x1x128 (int8)  | 7168 (int8)      | BuiltinOptionsType=0                                |
    |       |                 | 2 (int32)        |                  |                                                     |
    | 24    | fully_connected | 7168 (int8)      | 4 (int8)         | Activation:none                                     |
    |       |                 | 7168 (int8)      |                  |                                                     |
    |       |                 | 4 (int32)        |                  |                                                     |
    | 25    | softmax         | 4 (int8)         | 4 (int8)         | BuiltinOptionsType=9                                |
    +-------+-----------------+------------------+------------------+-----------------------------------------------------+
    Total MACs: 5.416 M
    Total OPs: 11.034 M
    Name: conv1d_example
    Version: 1
    Description: Conv1D example
    Classes: 1 NSR, 2 APB, 4 AFIB, 7 PVC
    hash: ba4d5f7fb808e7566d44a63b4335516e
    date: 2022-02-04T22:24:20.345Z
    runtime_memory_size: 33920
    samplewise_norm.rescale: 0.0
    samplewise_norm.mean_and_std: False
    fe.sample_rate_hz: 360
    fe.sample_length_ms: 10000
    fe.window_size_ms: 25
    fe.window_step_ms: 10
    fe.filterbank_n_channels: 32
    fe.filterbank_upper_band_limit: 7500.0
    fe.filterbank_lower_band_limit: 125.0
    fe.noise_reduction_enable: False
    fe.noise_reduction_smoothing_bits: 10
    fe.noise_reduction_even_smoothing: 0.02500000037252903
    fe.noise_reduction_odd_smoothing: 0.05999999865889549
    fe.noise_reduction_min_signal_remaining: 0.05000000074505806
    fe.pcan_enable: False
    fe.pcan_strength: 0.949999988079071
    fe.pcan_offset: 80.0
    fe.pcan_gain_bits: 21
    fe.log_scale_enable: True
    fe.log_scale_shift: 6
    fe.fft_length: 16
    .tflite file size: 82.8kB


Model Diagram
------------------

.. code-block:: shell
   
   mltk view conv1d_example --tflite --build

.. raw:: html

    <div class="model-diagram">
        <a href="../../../../_images/models/conv1d_example.tflite.png" target="_blank">
            <img src="../../../../_images/models/conv1d_example.tflite.png" />
            <p>Click to enlarge</p>
        </a>
    </div>

"""

import os
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras import regularizers

import mltk.core as mltk_core
from mltk.core.preprocess.audio.audio_feature_generator import AudioFeatureGeneratorSettings
from mltk.core.preprocess.audio.parallel_generator import ParallelAudioDataGenerator, ParallelProcessParams
from mltk.utils.archive_downloader import download_verify_extract


# Instantiate the model object with the following 'mixins':
# - TrainMixin        - Provides classifier model training operations and settings
# - AudioDatasetMixin - Provides audio data generation operations and settings
# - EvaluateClassifierMixin     - Provides classifier evaluation operations and settings
# @mltk_model # NOTE: This tag is required for this model be discoverable
class MyModel(
    mltk_core.MltkModel, 
    mltk_core.TrainMixin, 
    mltk_core.AudioDatasetMixin, 
    mltk_core.EvaluateClassifierMixin
):
    pass
my_model = MyModel()


#################################################
# General Settings
my_model.version = 1
my_model.description = 'Conv1D example'

#################################################
# Training Basic Settings
my_model.epochs = 40
my_model.batch_size = 5 
my_model.optimizer = 'adam'
my_model.metrics = ['accuracy']
my_model.loss = 'categorical_crossentropy'


#################################################
# Training callback Settings

my_model.checkpoint['monitor'] =  'val_accuracy'

# https://keras.io/api/callbacks/reduce_lr_on_plateau/
# my_model.reduce_lr_on_plateau['monitor'] = 'accuracy'
# my_model.reduce_lr_on_plateau['factor'] =  0.25
# my_model.reduce_lr_on_plateau['patience'] = 7

# # https://keras.io/api/callbacks/early_stopping/
# # If the validation accuracy doesn't improve after 'patience' epochs then stop training
# my_model.early_stopping['monitor'] = 'val_accuracy'
# my_model.early_stopping['patience'] = 20

def lr_schedule(epoch):
    initial_learning_rate = 0.001
    decay_per_epoch = 0.99
    lrate = initial_learning_rate * (decay_per_epoch ** epoch)
    print(f'Learning rate = {lrate}')
    return lrate

my_model.lr_schedule = lr_schedule


#################################################
# TF-Lite converter settings
my_model.tflite_converter['optimizations'] = ['DEFAULT']
my_model.tflite_converter['supported_ops'] = ['TFLITE_BUILTINS_INT8']
my_model.tflite_converter['inference_input_type'] = 'int8' # can also be float32
my_model.tflite_converter['inference_output_type'] = 'int8'
 # generate a representative dataset from the validation data
my_model.tflite_converter['representative_dataset'] = 'generate'


#################################################
# Audio Data Provider Settings

def load_dataset():
    DOWNLOAD_URL = 'https://github.com/SiliconLabs/mltk_assets/raw/master/datasets/ekg_waveforms.zip'
    VERIFY_SHA1 = '8221333C8ECEF29843D05B6385A4F47074820480'
    path = download_verify_extract(
        url=DOWNLOAD_URL,
        dest_subdir='datasets/ekg_waveforms/v1',
        file_hash=VERIFY_SHA1,
        show_progress=True,
        remove_root_dir=True
    )
    return path


my_model.dataset = load_dataset
my_model.class_mode = 'categorical'
my_model.classes = ['1 NSR','2 APB', '4 AFIB', '7 PVC']
validation_split = 0.1


sample_rate = 360
trim_threshold_db = 30
sample_length_seconds = 10.0


my_model.input_shape = (int(sample_rate*sample_length_seconds), 1) #Width, Channels



#################################################
# AudioFeatureGenerator Settings
# Even though the frontend is disabled
# We use this data structure to configure the parameters
# required by data augmentation
frontend_settings = AudioFeatureGeneratorSettings()
frontend_settings.sample_length_ms = sample_length_seconds*1000
frontend_settings.sample_rate_hz = sample_rate


#################################################
# ParallelAudioDataGenerator Settings

def post_processing_callback(params:ParallelProcessParams, x: np.ndarray):
    """
    X is an augmented sample as a 1D, float32 (-1., 1.) array
    Since the frontend is disabled, 
    we need to manually convert it to the data type and shape expected by the model
    NOTE: If you set the debug=True setting below, you can set a breakpoint here
    """
    x = np.reshape(x, my_model.input_shape)
    x *= 127
    assert params.dtype == np.int8
    return x.astype(params.dtype)


my_model.datagen = ParallelAudioDataGenerator(
    dtype=np.int8,
    frontend_enabled=False,
    frontend_settings=frontend_settings,
    sample_shape=my_model.input_shape, # Need to manually specify the sample shape since the frontend is disabled
    cores=0.5,
    debug=False, # Set this to true to enable debugging of the generator
    max_batches_pending=16, 
    validation_split= validation_split,
    validation_augmentation_enabled=True,
    samplewise_center=False,
    samplewise_std_normalization=False,
    rescale=None,
    unknown_class_percentage=0.0,
    silence_class_percentage=0.0,
    offset_range=(0.0,1.0),
    trim_threshold_db=trim_threshold_db,
    noise_colors=None,
    loudness_range=(0.8, 1.0),
    speed_range=(0.9,1.1),
    pitch_range=(0.9,1.1),
    bg_noise_range=(0.1,0.2),
    bg_noise_dir=None,
    postprocessing_function=post_processing_callback
)


#################################################
# Model Layout
def my_model_builder(model: MyModel):
    weight_decay = 1e-4
    regularizer = regularizers.l2(weight_decay)
    input_shape = model.input_shape
    filters = 16
 
    keras_model = Sequential([
        Conv1D(filters, 7, strides=4, padding='same', kernel_regularizer=regularizer, input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        #Dropout(rate=0.1),

        Conv1D(2*filters, 5, padding='same', kernel_regularizer=regularizer),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        #Dropout(rate=0.1),

        Conv1D(4*filters, 3, padding='same', kernel_regularizer=regularizer),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        #Dropout(rate=0.1),

        Conv1D(8*filters, 3, padding='same', kernel_regularizer=regularizer),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        #Dropout(rate=0.1),

        Flatten(),
        Dense(model.n_classes, activation='softmax')
    ])
 
    keras_model.compile(loss=model.loss, optimizer=model.optimizer, metrics=model.metrics)
    return keras_model


my_model.build_model_function = my_model_builder



##########################################################################################
# The following allows for running this model training script directly, e.g.: 
# python conv1d_example.py
#
# Note that this has the same functionality as:
# mltk train conv1d_example
#
if __name__ == '__main__':
    from mltk import cli

    # Setup the CLI logger
    cli.get_logger(verbose=False)

    # If this is true then this will do a "dry run" of the model testing
    # If this is false, then the model will be fully trained
    test_mode_enabled = True

    # Train the model
    # This does the same as issuing the command: mltk train conv1d_example-test --clean
    train_results = mltk_core.train_model(my_model, clean=True, test=test_mode_enabled)
    print(train_results)

    # Evaluate the model against the quantized .h5 (i.e. float32) model
    # This does the same as issuing the command: mltk evaluate conv1d_example-test
    tflite_eval_results = mltk_core.evaluate_model(my_model, verbose=True, test=test_mode_enabled)
    print(tflite_eval_results)

    # Profile the model in the simulator
    # This does the same as issuing the command: mltk profile conv1d_example-test
    profiling_results = mltk_core.profile_model(my_model, test=test_mode_enabled)
    print(profiling_results)