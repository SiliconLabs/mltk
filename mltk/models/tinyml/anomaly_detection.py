"""anomaly_detection
**********************

MLPerf Tiny anomaly detection reference model

- Source code: `anomaly_detection.py <https://github.com/siliconlabs/mltk/blob/master/mltk/models/tinyml/anomaly_detection.py>`_
- Pre-trained model: `anomaly_detection.mltk.zip <https://github.com/siliconlabs/mltk/blob/master/mltk/models/tinyml/anomaly_detection.mltk.zip>`_


Taken from:  
https://github.com/mlcommons/tiny/tree/master/benchmark/training/anomaly_detection

Additional information:  
https://github.com/SiliconLabs/platform_ml_models/tree/master/eembc/ToyADMOS_FC_AE


Dataset
--------
* https://zenodo.org/record/3351307
* See https://arxiv.org/pdf/1908.03299.pdf for details. Individual patterns were downloaded and used - page 2, figure 2.
* We have trained and tested the model with the toy car data, case 1 only. Code may be modified for more cases.
* By default the dataset is assumed to be located in `./ToyAdmos`.

Model Topology
--------------
* http://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds
* https://github.com/y-kawagu/dcase2020_task2_baseline

Spectrogram Characteristics
-----------------------------
* Front-end: `LIBROSA spectrogram <https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html>`_ is what is used in the code, 
  but we have also tested with https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/microfrontend
* Configuration: window=64ms, stride=32ms, bins=128, Upper frequency limit=24KHz, use only 5 center time windows

Performance (floating point model) 
----------------------------------
* Accuracy - 92.0%
* AUC - .923

Performance (quantized tflite model) 
--------------------------------------
* Accuracy - 91.5%
* AUC - .923



Commands
--------------

.. code-block:: console

   # Do a "dry run" test training of the model
   > mltk train anomaly_detection-test

   # Train the model
   > mltk train anomaly_detection

   # Evaluate the trained model .tflite model
   > mltk evaluate anomaly_detection --tflite

   # Profile the model in the MVP hardware accelerator simulator
   > mltk profile anomaly_detection --accelerator MVP

   # Profile the model on a physical development board
   > mltk profile anomaly_detection --accelerator MVP --device


Model Summary
--------------

.. code-block:: console
    
    > mltk summarize anomaly_detection --tflite
    
    +-------+-----------------+-------------------+-------------------+-----------------------+
    | Index | OpCode          | Input(s)          | Output(s)         | Config                |
    +-------+-----------------+-------------------+-------------------+-----------------------+
    | 0     | quantize        | 5x128x1 (float32) | 5x128x1 (int8)    | BuiltinOptionsType=0  |
    | 1     | reshape         | 5x128x1 (int8)    | 640 (int8)        | BuiltinOptionsType=0  |
    |       |                 | 2 (int32)         |                   |                       |
    | 2     | fully_connected | 640 (int8)        | 128 (int8)        | Activation:relu       |
    |       |                 | 640 (int8)        |                   |                       |
    |       |                 | 128 (int32)       |                   |                       |
    | 3     | fully_connected | 128 (int8)        | 128 (int8)        | Activation:relu       |
    |       |                 | 128 (int8)        |                   |                       |
    |       |                 | 128 (int32)       |                   |                       |
    | 4     | fully_connected | 128 (int8)        | 128 (int8)        | Activation:relu       |
    |       |                 | 128 (int8)        |                   |                       |
    |       |                 | 128 (int32)       |                   |                       |
    | 5     | fully_connected | 128 (int8)        | 128 (int8)        | Activation:relu       |
    |       |                 | 128 (int8)        |                   |                       |
    |       |                 | 128 (int32)       |                   |                       |
    | 6     | fully_connected | 128 (int8)        | 8 (int8)          | Activation:relu       |
    |       |                 | 128 (int8)        |                   |                       |
    |       |                 | 8 (int32)         |                   |                       |
    | 7     | fully_connected | 8 (int8)          | 128 (int8)        | Activation:relu       |
    |       |                 | 8 (int8)          |                   |                       |
    |       |                 | 128 (int32)       |                   |                       |
    | 8     | fully_connected | 128 (int8)        | 128 (int8)        | Activation:relu       |
    |       |                 | 128 (int8)        |                   |                       |
    |       |                 | 128 (int32)       |                   |                       |
    | 9     | fully_connected | 128 (int8)        | 128 (int8)        | Activation:relu       |
    |       |                 | 128 (int8)        |                   |                       |
    |       |                 | 128 (int32)       |                   |                       |
    | 10    | fully_connected | 128 (int8)        | 128 (int8)        | Activation:relu       |
    |       |                 | 128 (int8)        |                   |                       |
    |       |                 | 128 (int32)       |                   |                       |
    | 11    | fully_connected | 128 (int8)        | 640 (int8)        | Activation:none       |
    |       |                 | 128 (int8)        |                   |                       |
    |       |                 | 640 (int32)       |                   |                       |
    | 12    | shape           | 640 (int8)        | 2 (int32)         | BuiltinOptionsType=55 |
    | 13    | strided_slice   | 2 (int32)         | 0 (int32)         | BuiltinOptionsType=32 |
    |       |                 | 1 (int32)         |                   |                       |
    |       |                 | 1 (int32)         |                   |                       |
    |       |                 | 1 (int32)         |                   |                       |
    | 14    | pack            | 0 (int32)         | 4 (int32)         | BuiltinOptionsType=59 |
    |       |                 | 0 (int32)         |                   |                       |
    |       |                 | 0 (int32)         |                   |                       |
    |       |                 | 0 (int32)         |                   |                       |
    | 15    | reshape         | 640 (int8)        | 5x128x1 (int8)    | BuiltinOptionsType=0  |
    |       |                 | 4 (int32)         |                   |                       |
    | 16    | dequantize      | 5x128x1 (int8)    | 5x128x1 (float32) | BuiltinOptionsType=0  |
    +-------+-----------------+-------------------+-------------------+-----------------------+
    Total MACs: 264.192 k
    Total OPs: 535.176 k
    Name: anomaly_detection
    Version: 1
    Description: TinyML: Anonomly Detection - Fully Connect AutoEncoder with ToyADMOS
    Classes: normal
    hash: 5cf2dc0ea093044c7a31a226d44b8084
    date: 2022-02-04T19:15:48.676Z
    runtime_memory_size: 9396
    samplewise_norm.rescale: 0.0
    samplewise_norm.mean_and_std: False
    .tflite file size: 280.3kB


Model Diagram
------------------

.. code-block:: console
   
   > mltk view anomaly_detection --tflite

.. raw:: html

    <div class="model-diagram">
        <a href="../../../../_images/models/anomaly_detection.tflite.png" target="_blank">
            <img src="../../../../_images/models/anomaly_detection.tflite.png" />
            <p>Click to enlarge</p>
        </a>
    </div>

"""

import random as rnd
import numpy as np
from mltk.core.preprocess.image.parallel_generator import ParallelImageDataGenerator, ParallelProcessParams
from mltk.core.model import (
    MltkModel,
    TrainMixin,
    ImageDatasetMixin,
    EvaluateAutoEncoderMixin
)
from mltk.models.shared import FullyConnectedAutoEncoder
from mltk.utils.archive_downloader import download_verify_extract


# Instantiate the MltkModel object with the following 'mixins':
# - TrainMixin            - Provides classifier model training operations and settings
# - ImageDatasetMixin     - Provides image data generation operations and settings
# - EvaluateAutoEncoderMixin  - Provides auto-encoder evaluation operations and settings
# @mltk_model   # NOTE: This tag is required for this model be discoverable
class MyModel(
    MltkModel, 
    TrainMixin, 
    ImageDatasetMixin, 
    EvaluateAutoEncoderMixin
):
    pass
my_model = MyModel()

# General parameters
my_model.version = 1
my_model.description = 'TinyML: Anonomly Detection - Fully Connect AutoEncoder with ToyADMOS'


#################################################
# Training parameters
my_model.epochs = 100
my_model.batch_size = 40 
my_model.optimizer = 'adam'
my_model.loss = 'mean_squared_error'
my_model.metrics = ['mean_squared_error']


#################################################
# Image Dataset Settings

# The directory of the training data
def download_dataset():
    return download_verify_extract(
        url='https://github.com/SiliconLabs/mltk_assets/raw/master/datasets/features_toy_car_all.7z',
        dest_subdir='datasets/toyadmos/preprocessed/v1',
        file_hash='8FC5779A38694EB17E75AD21EF457BD39E6EF937',
        show_progress=True,
        remove_root_dir=True
    )


my_model.dataset = download_dataset
# Auto-Encoder directly pass the image into model
my_model.class_mode = 'input'
# The class labels found in your training dataset directory
my_model.classes = ['normal']
# Don't use weights for auto-encoders
my_model.class_weights = None
# The input shape to the model. The dataset samples will be resized if necessary
# This is the shape defined by the ToyADMOS dataset
my_model.input_shape = (5, 128, 1)
# We manually re-shape the images in the reshape_input_callback() function below
my_model.interpolation = None

validation_split = 0.1


##############################################################
# Training callbacks
#

my_model.checkpoint['monitor'] = 'val_loss'
my_model.checkpoint['mode'] = 'auto'



##############################################################
# Image data generator settings
#
def reshape_input_callback(params: ParallelProcessParams, x: np.ndarray):
    input_shape = my_model.input_shape
    input_height = input_shape[0]
    x_length = x.shape[0]

    # Determine the beginning of the subsection of the input sample
    # we will use for training or validation
    if params.subset == 'training':
        # If we're training, the get a random subsection of the input data
        left_offset = 50
        right_offset = 50
        start = left_offset + rnd.randrange((x_length - input_height+1) - (left_offset + right_offset))
    else:
        # Otherwise, for validation/evaluation always start at the middle of the input
        start = int(x_length/2)

    # Get the input subsection
    subsection_data = np.array(x[start: start+input_height, :])
    # and reshape to the expected shape of the model
    y = subsection_data.reshape(input_shape)
    
    return y


my_model.datagen = ParallelImageDataGenerator(
    cores=.35,
    max_batches_pending=16,
    debug=False, # set to True to enable debuging the reshape_input_callback() callback
    validation_split=validation_split,
    preprocessing_function = reshape_input_callback,
)


##############################################################
# Model Layout
def my_model_builder(model: MyModel):
    autoencoder = FullyConnectedAutoEncoder(
        input_shape=model.input_shape
    )
    autoencoder.compile(
        loss=model.loss, 
        optimizer=model.optimizer,
        metrics=model.metrics
    )
    return autoencoder

my_model.build_model_function = my_model_builder