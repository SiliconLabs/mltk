"""autoencoder_example
************************

- Source code: `autoencoder_example.py <https://github.com/siliconlabs/mltk/blob/master/mltk/models/examples/autoencoder_example.py>`_
- Pre-trained model: `autoencoder_example.mltk.zip <https://github.com/siliconlabs/mltk/blob/master/mltk/models/examples/autoencoder_example.mltk.zip>`_


This demonstrates how to build an autoencoder model.
This is based on `Tensorflow: Anomaly detection <https://www.tensorflow.org/tutorials/generative/autoencoder#third_example_anomaly_detection>`_ 

In this example, you will train an autoencoder to detect anomalies on the `ECG5000 <http://www.timeseriesclassification.com/description.php?Dataset=ECG5000>`_ dataset. 
This dataset contains 5,000 `Electrocardiograms <https://en.wikipedia.org/wiki/Electrocardiography>`_, each with 140 data points. You will use a simplified version of the dataset, 
where each example has been labeled either 0 (corresponding to an abnormal rhythm), or 1 (corresponding to a normal rhythm). 
You are interested in identifying the abnormal rhythms.


Commands
--------------

.. code-block:: shell

   # Do a "dry run" test training of the model
   mltk train autoencoder_example-test

   # Train the model
   mltk train autoencoder_example

   # Evaluate the trained model .tflite model
   # Also dump a comparsion of the original image vs the generated autoencoder image
   mltk evaluate autoencoder_example --tflite --dump -- count 20

   # Profile the model in the MVP hardware accelerator simulator
   mltk profile autoencoder_example --accelerator MVP

   # Profile the model on a physical development board
   mltk profile autoencoder_example --accelerator MVP --device

   # Directly invoke the model script
   python autoencoder_example.py

Model Summary
--------------

.. code-block:: shell
    
    mltk summarize autoencoder_example --tflite
    
    +-------+-----------------+---------------+---------------+----------------------+ 
    | Index | OpCode          | Input(s)      | Output(s)     | Config               | 
    +-------+-----------------+---------------+---------------+----------------------+ 
    | 0     | quantize        | 140 (float32) | 140 (int8)    | BuiltinOptionsType=0 | 
    | 1     | fully_connected | 140 (int8)    | 32 (int8)     | Activation:relu      | 
    |       |                 | 140 (int8)    |               |                      | 
    |       |                 | 32 (int32)    |               |                      | 
    | 2     | fully_connected | 32 (int8)     | 16 (int8)     | Activation:relu      | 
    |       |                 | 32 (int8)     |               |                      | 
    |       |                 | 16 (int32)    |               |                      | 
    | 3     | fully_connected | 16 (int8)     | 8 (int8)      | Activation:relu      | 
    |       |                 | 16 (int8)     |               |                      | 
    |       |                 | 8 (int32)     |               |                      | 
    | 4     | fully_connected | 8 (int8)      | 16 (int8)     | Activation:relu      | 
    |       |                 | 8 (int8)      |               |                      | 
    |       |                 | 16 (int32)    |               |                      | 
    | 5     | fully_connected | 16 (int8)     | 32 (int8)     | Activation:relu      | 
    |       |                 | 16 (int8)     |               |                      | 
    |       |                 | 32 (int32)    |               |                      | 
    | 6     | fully_connected | 32 (int8)     | 140 (int8)    | Activation:none      | 
    |       |                 | 32 (int8)     |               |                      | 
    |       |                 | 140 (int32)   |               |                      | 
    | 7     | logistic        | 140 (int8)    | 140 (int8)    | BuiltinOptionsType=0 | 
    | 8     | dequantize      | 140 (int8)    | 140 (float32) | BuiltinOptionsType=0 | 
    +-------+-----------------+---------------+---------------+----------------------+ 
    Total MACs: 10.240 k                                                               
    Total OPs: 21.564 k                                                                
    Name: autoencoder_example                                                          
    Version: 1                                                                         
    Description: Autoencoder example to detect anomalies in ECG dataset                
    classes: []                                                                        
    hash: 66c8e81181a47dfcc2f0ff53a55aef49                                             
    date: 2022-04-28T19:08:38.662Z                                                     
    runtime_memory_size: 2028                                                          
    .tflite file size: 15.8kB                                                          


Model Profiling Report
-----------------------

.. code-block:: shell
   
   # Profile on physical EFR32xG24 using MVP accelerator
   mltk profile autoencoder_example --device --accelerator MVP

    Profiling Summary
    Name: autoencoder_example
    Accelerator: MVP
    Input Shape: 1x140
    Input Data Type: float32
    Output Shape: 1x140
    Output Data Type: float32
    Flash, Model File Size (bytes): 15.7k
    RAM, Runtime Memory Size (bytes): 3.4k
    Operation Count: 21.8k
    Multiply-Accumulate Count: 10.2k
    Layer Count: 9
    Unsupported Layer Count: 0
    Accelerator Cycle Count: 16.9k
    CPU Cycle Count: 131.5k
    CPU Utilization (%): 89.2
    Clock Rate (hz): 78.0M
    Time (s): 1.9m
    Ops/s: 11.5M
    MACs/s: 5.4M
    Inference/s: 529.1

    Model Layers
    +-------+-----------------+-------+--------+------------+------------+----------+-----------------+--------------+-----------------+
    | Index | OpCode          | # Ops | # MACs | Acc Cycles | CPU Cycles | Time (s) | Input Shape     | Output Shape | Options         |
    +-------+-----------------+-------+--------+------------+------------+----------+-----------------+--------------+-----------------+
    | 0     | quantize        | 560.0 | 0      | 0          | 5.5k       | 90.0u    | 1x140           | 1x140        | Type=none       |
    | 1     | fully_connected | 9.1k  | 4.5k   | 6.9k       | 2.3k       | 120.0u   | 1x140,32x140,32 | 1x32         | Activation:relu |
    | 2     | fully_connected | 1.1k  | 512.0  | 878.0      | 1.9k       | 30.0u    | 1x32,16x32,16   | 1x16         | Activation:relu |
    | 3     | fully_connected | 280.0 | 128.0  | 254.0      | 1.9k       | 30.0u    | 1x16,8x16,8     | 1x8          | Activation:relu |
    | 4     | fully_connected | 304.0 | 128.0  | 302.0      | 1.9k       | 30.0u    | 1x8,16x8,16     | 1x16         | Activation:relu |
    | 5     | fully_connected | 1.1k  | 512.0  | 974.0      | 1.9k       | 30.0u    | 1x16,32x16,32   | 1x32         | Activation:relu |
    | 6     | fully_connected | 9.1k  | 4.5k   | 7.6k       | 1.9k       | 120.0u   | 1x32,140x32,140 | 1x140        | Activation:none |
    | 7     | logistic        | 0     | 0      | 0          | 96.0k      | 1.2m     | 1x140           | 1x140        | Type=none       |
    | 8     | dequantize      | 280.0 | 0      | 0          | 18.0k      | 210.0u   | 1x140           | 1x140        | Type=none       |
    +-------+-----------------+-------+--------+------------+------------+----------+-----------------+--------------+-----------------+


Model Diagram
------------------

.. code-block:: shell
   
   mltk view autoencoder_example --tflite

.. raw:: html

    <div class="model-diagram">
        <a href="../../../../_images/models/autoencoder_example.tflite.png" target="_blank">
            <img src="../../../../_images/models/autoencoder_example.tflite.png" />
            <p>Click to enlarge</p>
        </a>
    </div>


"""
from typing import List
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


import mltk.core as mltk_core
from mltk.utils.path import create_user_dir
from mltk.utils.archive_downloader import download_url



# Instantiate the MltkModel object with the following 'mixins':
# - TrainMixin            - Provides classifier model training operations and settings
# - DatasetMixin          - Provides general dataset operations and settings
# - EvaluateClassifierMixin         - Provides classifier evaluation operations and settings
# @mltk_model # NOTE: This tag is required for this model be discoverable
class MyModel(
    mltk_core.MltkModel, 
    mltk_core.TrainMixin, 
    mltk_core.DatasetMixin, 
    mltk_core.EvaluateAutoEncoderMixin
):
    def load_dataset(
        self, 
        subset: str,  
        classes:List[str]=None,
        max_samples_per_class=None,
        test:bool=False,
        **kwargs
    ):
        super().load_dataset(subset) 

        if test:
            max_samples_per_class = 3

        # Download the dataset (if necessary)
        dataset_path = f'{create_user_dir()}/datasets/ecg500.csv'
        download_url(
            'http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv',
            dataset_path
        )

        # Load the dataset into numpy array
        dataset = np.genfromtxt(dataset_path, delimiter=',', dtype=np.float32)

        # The last column contains the labels
        labels = dataset[:, -1]
        data = dataset[:,:-1]

        # Split the data into training and test data
        self.validation_split = 0.2
        train_data, test_data, train_labels, test_labels = train_test_split(
            data, labels, test_size=self.validation_split, random_state=21
        )

        min_val = tf.reduce_min(train_data)
        max_val = tf.reduce_max(train_data)

        train_data = (train_data - min_val) / (max_val - min_val)
        test_data = (test_data - min_val) / (max_val - min_val)

        train_labels_bool = train_labels.astype(bool)
        test_labels_bool = test_labels.astype(bool)

        normal_train_data = train_data[train_labels_bool]
        normal_test_data = test_data[test_labels_bool]

        anomalous_train_data = train_data[~train_labels_bool]
        anomalous_test_data = test_data[~test_labels_bool]

        self._normal_train_count = len(normal_train_data)
        self._normal_test_count = len(normal_test_data)
        self._abnormal_train_count = len(anomalous_train_data)
        self._abnormal_test_count = len(anomalous_test_data)

        # If we're evaluating,
        # then just return the "normal" or "abnormal" samples
        # NOTE: The y value is not required in this case
        if subset == 'evaluation':
            
            if classes[0] =='normal':
                x = normal_test_data
            else:
                x = anomalous_test_data

            if max_samples_per_class:
                sample_count = min(len(x), max_samples_per_class)
                x = x[0:sample_count]
            self.x = x
        else:
            # For training, we just use the "normal" data
            # Note that x and y use the same data as the whole point 
            #  of an autoencoder is to reconstruct the input data
            self.x = normal_train_data
            self.y = normal_train_data
            self.validation_data = (test_data, test_data)


    def summarize_dataset(self) -> str: 
        s = f'Train dataset: Found {self._normal_train_count} "normal", {self._abnormal_train_count} "abnormal" samples\n'
        s += f'Validation dataset: Found {self._normal_test_count} "normal", {self._abnormal_test_count} "abnormal" samples'
        return s




my_model = MyModel()


#################################################
# General Settings
# 
my_model.version = 1
my_model.description = 'Autoencoder example to detect anomalies in ECG dataset'

my_model.input_shape = (140,)

#################################################
# Training Settings
my_model.epochs = 20
my_model.batch_size = 512
my_model.optimizer = 'adam'
my_model.metrics = ['mae']
my_model.loss = 'mae'

#################################################
# Training callback Settings

# Generate a training weights .h5 whenever the 
# val_accuracy improves
my_model.checkpoint['monitor'] =  'val_loss'
my_model.checkpoint['mode'] =  'auto'


#################################################
# TF-Lite converter settings
my_model.tflite_converter['optimizations'] = ['DEFAULT']
my_model.tflite_converter['supported_ops'] = ['TFLITE_BUILTINS_INT8']
my_model.tflite_converter['inference_input_type'] = tf.float32
my_model.tflite_converter['inference_output_type'] = tf.float32
 # generate a representative dataset from the validation data
my_model.tflite_converter['representative_dataset'] = 'generate'




#################################################
# Build the ML Model
def my_model_builder(model: MyModel):
    model_input = tf.keras.layers.Input(shape=model.input_shape)
    encoder = tf.keras.Sequential([
        model_input,
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu")]
    )

    decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(140, activation="sigmoid")
    ])

    autoencoder = tf.keras.models.Model(model_input, decoder(encoder(model_input)))
    autoencoder.compile(
        loss=model.loss, 
        optimizer=model.optimizer, 
        metrics=model.metrics
    )

    return autoencoder

my_model.build_model_function = my_model_builder




##########################################################################################
# The following allows for running this model training script directly, e.g.: 
# python autoencoder_example.py
#
# Note that this has the same functionality as:
# mltk train autoencoder_example
#
if __name__ == '__main__':
    from mltk import cli

    # Setup the CLI logger
    cli.get_logger(verbose=False)

    # If this is true then this will do a "dry run" of the model testing
    # If this is false, then the model will be fully trained
    test_mode_enabled = True

    # Train the model
    # This does the same as issuing the command: mltk train autoencoder_example-test --clean
    train_results = mltk_core.train_model(my_model, clean=True, test=test_mode_enabled)
    print(train_results)

    # Evaluate the model against the quantized .h5 (i.e. float32) model
    # This does the same as issuing the command: mltk evaluate autoencoder_example-test
    tflite_eval_results = mltk_core.evaluate_model(my_model, verbose=True, test=test_mode_enabled)
    print(tflite_eval_results)

    # Profile the model in the simulator
    # This does the same as issuing the command: mltk profile autoencoder_example-test
    profiling_results = mltk_core.profile_model(my_model, test=test_mode_enabled)
    print(profiling_results)