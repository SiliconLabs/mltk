"""tflite_micro_magic_wand
*******************************

- Source code: `tflite_micro_magic_wand.py <https://github.com/siliconlabs/mltk/blob/master/mltk/models/tflite_micro/tflite_micro_magic_wand.py>`_
- Pre-trained model: `tflite_micro_magic_wand.mltk.zip <https://github.com/siliconlabs/mltk/blob/master/mltk/models/tflite_micro/tflite_micro_magic_wand.mltk.zip>`_


https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/magic_wand




Commands
--------------

.. code-block:: shell

   # Do a "dry run" test training of the model
   mltk train tflite_micro_magic_wand-test

   # Train the model
   mltk train tflite_micro_magic_wand

   # Evaluate the trained model .tflite model
   mltk evaluate tflite_micro_magic_wand --tflite

   # Profile the model in the MVP hardware accelerator simulator
   mltk profile tflite_micro_magic_wand --accelerator MVP

   # Profile the model on a physical development board
   mltk profile tflite_micro_magic_wand --accelerator MVP --device



Model Summary
--------------

.. code-block:: shell
    
    mltk summarize tflite_micro_magic_wand --tflite
    
    +-------+-----------------+-------------------+----------------+-----------------------------------------------------+
    | Index | OpCode          | Input(s)          | Output(s)      | Config                                              |
    +-------+-----------------+-------------------+----------------+-----------------------------------------------------+
    | 0     | quantize        | 128x3x1 (float32) | 128x3x1 (int8) | BuiltinOptionsType=0                                |
    | 1     | conv_2d         | 128x3x1 (int8)    | 128x3x8 (int8) | Padding:same stride:1x1 activation:relu             |
    |       |                 | 4x3x1 (int8)      |                |                                                     |
    |       |                 | 8 (int32)         |                |                                                     |
    | 2     | max_pool_2d     | 128x3x8 (int8)    | 42x1x8 (int8)  | Padding:valid stride:3x3 filter:3x3 activation:none |
    | 3     | conv_2d         | 42x1x8 (int8)     | 42x1x16 (int8) | Padding:same stride:1x1 activation:relu             |
    |       |                 | 4x1x8 (int8)      |                |                                                     |
    |       |                 | 16 (int32)        |                |                                                     |
    | 4     | max_pool_2d     | 42x1x16 (int8)    | 14x1x16 (int8) | Padding:same stride:1x3 filter:1x3 activation:none  |
    | 5     | reshape         | 14x1x16 (int8)    | 224 (int8)     | BuiltinOptionsType=0                                |
    |       |                 | 2 (int32)         |                |                                                     |
    | 6     | fully_connected | 224 (int8)        | 16 (int8)      | Activation:relu                                     |
    |       |                 | 224 (int8)        |                |                                                     |
    |       |                 | 16 (int32)        |                |                                                     |
    | 7     | fully_connected | 16 (int8)         | 4 (int8)       | Activation:none                                     |
    |       |                 | 16 (int8)         |                |                                                     |
    |       |                 | 4 (int32)         |                |                                                     |
    | 8     | softmax         | 4 (int8)          | 4 (int8)       | BuiltinOptionsType=9                                |
    | 9     | dequantize      | 4 (int8)          | 4 (float32)    | BuiltinOptionsType=0                                |
    +-------+-----------------+-------------------+----------------+-----------------------------------------------------+
    Total MACs: 62.016 k
    Total OPs: 133.268 k
    Name: tflite_micro_magic_wand
    Version: 1
    Description: TFLite-Micro Magic Wand
    Classes: wing, ring, slope, negative
    hash: c044953d468755c572f05f4f2750d1ef
    date: 2022-02-04T19:11:55.646Z
    runtime_memory_size: 5444
    .tflite file size: 10.2kB


Model Profiling Report
-----------------------

.. code-block:: shell
   
   # Profile on physical EFR32xG24 using MVP accelerator
   mltk profile tflite_micro_magic_wand --device --accelerator MVP

    Profiling Summary
    Name: tflite_micro_magic_wand
    Accelerator: MVP
    Input Shape: 1x128x3x1
    Input Data Type: float32
    Output Shape: 1x4
    Output Data Type: float32
    Flash, Model File Size (bytes): 10.2k
    RAM, Runtime Memory Size (bytes): 6.4k
    Operation Count: 140.6k
    Multiply-Accumulate Count: 62.0k
    Layer Count: 10
    Unsupported Layer Count: 0
    Accelerator Cycle Count: 72.9k
    CPU Cycle Count: 94.6k
    CPU Utilization (%): 62.2
    Clock Rate (hz): 78.0M
    Time (s): 2.0m
    Ops/s: 72.1M
    MACs/s: 31.8M
    Inference/s: 512.8

    Model Layers
    +-------+-----------------+-------+--------+------------+------------+----------+----------------------+--------------+-----------------------------------------------------+
    | Index | OpCode          | # Ops | # MACs | Acc Cycles | CPU Cycles | Time (s) | Input Shape          | Output Shape | Options                                             |
    +-------+-----------------+-------+--------+------------+------------+----------+----------------------+--------------+-----------------------------------------------------+
    | 0     | quantize        | 1.5k  | 0      | 0          | 13.9k      | 180.0u   | 1x128x3x1            | 1x128x3x1    | Type=none                                           |
    | 1     | conv_2d         | 82.9k | 36.9k  | 46.4k      | 32.4k      | 900.0u   | 1x128x3x1,8x4x3x1,8  | 1x128x3x8    | Padding:same stride:1x1 activation:relu             |
    | 2     | max_pool_2d     | 3.0k  | 0      | 1.9k       | 8.5k       | 120.0u   | 1x128x3x8            | 1x42x1x8     | Padding:valid stride:3x3 filter:3x3 activation:none |
    | 3     | conv_2d         | 45.0k | 21.5k  | 18.4k      | 13.2k      | 360.0u   | 1x42x1x8,16x4x1x8,16 | 1x42x1x16    | Padding:same stride:1x1 activation:relu             |
    | 4     | max_pool_2d     | 672.0 | 0      | 672.0      | 15.6k      | 210.0u   | 1x42x1x16            | 1x14x1x16    | Padding:same stride:1x3 filter:1x3 activation:none  |
    | 5     | reshape         | 0     | 0      | 0          | 1.6k       | 0        | 1x14x1x16,2          | 1x224        | Type=none                                           |
    | 6     | fully_connected | 7.2k  | 3.6k   | 5.5k       | 2.2k       | 120.0u   | 1x224,16x224,16      | 1x16         | Activation:relu                                     |
    | 7     | fully_connected | 132.0 | 64.0   | 135.0      | 1.9k       | 30.0u    | 1x16,4x16,4          | 1x4          | Activation:none                                     |
    | 8     | softmax         | 20.0  | 0      | 0          | 4.1k       | 30.0u    | 1x4                  | 1x4          | Type=softmaxoptions                                 |
    | 9     | dequantize      | 8.0   | 0      | 0          | 1.2k       | 0        | 1x4                  | 1x4          | Type=none                                           |
    +-------+-----------------+-------+--------+------------+------------+----------+----------------------+--------------+-----------------------------------------------------+


Model Diagram
------------------

.. code-block:: shell
   
   mltk view tflite_micro_magic_wand --tflite

.. raw:: html

    <div class="model-diagram">
        <a href="../../../../_images/models/tflite_micro_magic_wand.tflite.png" target="_blank">
            <img src="../../../../_images/models/tflite_micro_magic_wand.tflite.png" />
            <p>Click to enlarge</p>
        </a>
    </div>




"""
# pylint: disable=attribute-defined-outside-init


import tensorflow as tf
import numpy as np

from mltk.core.model import (
    MltkModel,
    TrainMixin,
    DatasetMixin,
    EvaluateClassifierMixin
)

from mltk.datasets.accelerometer import tflm_magic_wand as tflm_magic_wand_dataset

# Instantiate the model object with the following 'mixins':
# - TrainMixin        - Provides classifier model training operations and settings
# - DatasetMixin      - Provides generic dataset properties
# - EvaluateClassifierMixin     - Provides classifier evaluation operations and settings
# @mltk_model # NOTE: This tag is required for this model be discoverable
class MyModel(
    MltkModel, 
    TrainMixin, 
    DatasetMixin, 
    EvaluateClassifierMixin
):
    def load_dataset(
        self, 
        subset: str,  
        test:bool = False,
        **kwargs
    ):
        super().load_dataset(subset) 

        batch_size = self.batch_size
        seq_length = self.input_shape[0]
        self._data_loader = tflm_magic_wand_dataset.load_data(seq_length=seq_length)

        self._data_loader.format()  
        train_data = self._data_loader.train_data 
        validation_data = self._data_loader.valid_data
        test_data = self._data_loader.test_data

        def reshape_function(data, label):
            reshaped_data = tf.reshape(data, [-1, 3, 1])
            return reshaped_data, label

        train_data = train_data.map(reshape_function)
        validation_data = validation_data.map(reshape_function)
        test_data = test_data.map(reshape_function)


        if subset == 'evaluation':
            self.x = test_data.batch(batch_size)
            self.eval_steps_per_epoch = int((len(test_data) - 1) / batch_size + 1)
        else:
            self.x = train_data.batch(batch_size).repeat()
            self.validation_data = validation_data.batch(batch_size)
            self.validation_steps = int((len(validation_data) - 1) / batch_size + 1)
            # This is needed so we can generate representative data
            # for quantization
            self._unbatched_validation_data = validation_data


    def summarize_dataset(self) -> str: 
        train_data = self._data_loader.train_data 
        validation_data = self._data_loader.valid_data
        s = f'Train dataset: Found {len(train_data)} samples belonging to 4 classes\n'
        s += f'Validation dataset: Found {len(validation_data)} samples belonging to 4 classes'

        return s


my_model = MyModel()


#################################################
# General Settings
my_model.version = 1
my_model.description = 'TFLite-Micro Magic Wand'


#################################################
# Training Basic Settings
my_model.epochs = 50
my_model.steps_per_epoch = 1000
my_model.batch_size = 64 
my_model.optimizer = 'adam'
my_model.metrics = ['accuracy']
my_model.loss = 'sparse_categorical_crossentropy'


#################################################
# TF-Lite converter settings
my_model.tflite_converter['optimizations'] = ['DEFAULT']
my_model.tflite_converter['supported_ops'] = ['TFLITE_BUILTINS_INT8']
my_model.tflite_converter['inference_input_type'] = tf.float32
my_model.tflite_converter['inference_output_type'] =  tf.float32
# generate a representative dataset from the validation data
def _generate_representative_dataset():
    retval = []
    for x, _ in  my_model._unbatched_validation_data.batch(1).take(100): # pylint: disable=protected-access
        retval.append([x])
    return retval

my_model.tflite_converter['representative_dataset'] = _generate_representative_dataset


#################################################
# Dataset Settings
my_model.input_shape = (128, 3, 1)
my_model.classes = ('wing', 'ring', 'slope', 'negative')
my_model.class_weights = 'none'


#################################################
# Model Layout
def my_model_builder(model: MyModel):
    keras_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            8, (4, 3),
            padding="same",
            activation="relu",
            input_shape=model.input_shape),  # output_shape=(batch, 128, 3, 8)
        tf.keras.layers.MaxPool2D((3, 3)),  # (batch, 42, 1, 8)
        tf.keras.layers.Dropout(0.1),  # (batch, 42, 1, 8)
        tf.keras.layers.Conv2D(16, (4, 1), padding="same",
                                activation="relu"),  # (batch, 42, 1, 16)
        tf.keras.layers.MaxPool2D((3, 1), padding="same"),  # (batch, 14, 1, 16)
        tf.keras.layers.Dropout(0.1),  # (batch, 14, 1, 16)
        tf.keras.layers.Flatten(),  # (batch, 224)
        tf.keras.layers.Dense(16, activation="relu"),  # (batch, 16)
        tf.keras.layers.Dropout(0.1),  # (batch, 16)
        tf.keras.layers.Dense(model.n_classes, activation="softmax")  # (batch, 4)
    ], name=model.name)
    keras_model.compile(
        loss=model.loss, 
        optimizer=model.optimizer, 
        metrics=model.metrics
    )
    return keras_model

my_model.build_model_function = my_model_builder



##########################################################################################
# The following allows for running this model training script directly, e.g.: 
# python tflite_micro_magic_wand.py
#
# Note that this has the same functionality as:
# mltk train tflite_micro_magic_wand
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
    # This does the same as issuing the command: mltk train tflite_micro_magic_wand-test --clean
    train_results = mltk_core.train_model(my_model, clean=True, test=test_mode_enabled)
    print(train_results)

    # Evaluate the model against the quantized .h5 (i.e. float32) model
    # This does the same as issuing the command: mltk evaluate tflite_micro_magic_wand-test
    tflite_eval_results = mltk_core.evaluate_model(my_model, verbose=True, test=test_mode_enabled)
    print(tflite_eval_results)

    # Profile the model in the simulator
    # This does the same as issuing the command: mltk profile tflite_micro_magic_wand-test
    profiling_results = mltk_core.profile_model(my_model, test=test_mode_enabled)
    print(profiling_results)