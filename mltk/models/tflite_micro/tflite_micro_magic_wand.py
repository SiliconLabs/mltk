"""tflite_micro_magic_wand
*******************************

- Source code: `tflite_micro_magic_wand.py <https://github.com/siliconlabs/mltk/blob/master/mltk/models/tflite_micro/tflite_micro_magic_wand.py>`_
- Pre-trained model: `tflite_micro_magic_wand.mltk.zip <https://github.com/siliconlabs/mltk/blob/master/mltk/models/tflite_micro/tflite_micro_magic_wand.mltk.zip>`_


https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/magic_wand




Commands
--------------

.. code-block:: console

   # Do a "dry run" test training of the model
   > mltk train tflite_micro_magic_wand-test

   # Train the model
   > mltk train tflite_micro_magic_wand

   # Evaluate the trained model .tflite model
   > mltk evaluate tflite_micro_magic_wand --tflite

   # Profile the model in the MVP hardware accelerator simulator
   > mltk profile tflite_micro_magic_wand --accelerator MVP

   # Profile the model on a physical development board
   > mltk profile tflite_micro_magic_wand --accelerator MVP --device



Model Summary
--------------

.. code-block:: console
    
    > mltk summarize tflite_micro_magic_wand --tflite
    
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



Model Diagram
------------------

.. code-block:: console
   
   > mltk view tflite_micro_magic_wand --tflite

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
            # Convert the  test "y" values to a numpy array
            # which is required by the eval scripts
            self.y = np.zeros((self.eval_steps_per_epoch*batch_size, ), dtype=np.int32)
            for batch_id, batch in enumerate(test_data.batch(batch_size).take(self.eval_steps_per_epoch)):
                batch_offset = batch_id * batch_size
                self.y[batch_offset: batch_offset + batch_size] = batch[1].numpy()

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