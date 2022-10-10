"""image_example1
******************

- Source code: `image_example1.py <https://github.com/siliconlabs/mltk/blob/master/mltk/models/examples/image_example1.py>`_
- Pre-trained model: `image_example1.mltk.zip <https://github.com/siliconlabs/mltk/blob/master/mltk/models/examples/image_example1.mltk.zip>`_

This provides an example of how to define a classification model 
that uses the Rock/Paper/Scissors dataset with the ParallelImageGenerator as its data source.

The basic flow for the ML model is:

``96x96x1 grayscale image of hand gesture -> ML Model -> [result vector]``


Where `[result vector]` is a 3 element array with each element containing the % probability that the 
given image is a "rock", "paper", or "scissor" hand gesture.



Commands
--------------

.. code-block:: shell

   # Do a "dry run" test training of the model
   mltk train image_example1-test

   # Train the model
   mltk train image_example1

   # Evaluate the trained model .tflite model
   mltk evaluate image_example1 --tflite

   # Profile the model in the MVP hardware accelerator simulator
   mltk profile image_example1 --accelerator MVP

   # Profile the model on a physical development board
   mltk profile image_example1 --accelerator MVP --device

   # Use the model custom command to dump the augmented samples to ~/.mltk/models/image_example1/datagen_dump
   mltk custom image_example1 datagen_dump --count 20

   # Directly invoke the model script
   python image_example1.py



Model Summary
--------------

.. code-block:: shell
    
    mltk summarize image_example1 --tflite
    
    +-------+-----------------+-------------------+-----------------+-----------------------------------------------------+
    | Index | OpCode          | Input(s)          | Output(s)       | Config                                              |
    +-------+-----------------+-------------------+-----------------+-----------------------------------------------------+
    | 0     | quantize        | 96x96x1 (float32) | 96x96x1 (int8)  | BuiltinOptionsType=0                                |
    | 1     | conv_2d         | 96x96x1 (int8)    | 48x48x24 (int8) | Padding:same stride:2x2 activation:relu             |
    |       |                 | 3x3x1 (int8)      |                 |                                                     |
    |       |                 | 24 (int32)        |                 |                                                     |
    | 2     | average_pool_2d | 48x48x24 (int8)   | 24x24x24 (int8) | Padding:valid stride:2x2 filter:2x2 activation:none |
    | 3     | conv_2d         | 24x24x24 (int8)   | 11x11x16 (int8) | Padding:valid stride:2x2 activation:relu            |
    |       |                 | 3x3x24 (int8)     |                 |                                                     |
    |       |                 | 16 (int32)        |                 |                                                     |
    | 4     | conv_2d         | 11x11x16 (int8)   | 9x9x24 (int8)   | Padding:valid stride:1x1 activation:relu            |
    |       |                 | 3x3x16 (int8)     |                 |                                                     |
    |       |                 | 24 (int32)        |                 |                                                     |
    | 5     | average_pool_2d | 9x9x24 (int8)     | 4x4x24 (int8)   | Padding:valid stride:2x2 filter:2x2 activation:none |
    | 6     | reshape         | 4x4x24 (int8)     | 384 (int8)      | BuiltinOptionsType=0                                |
    |       |                 | 2 (int32)         |                 |                                                     |
    | 7     | fully_connected | 384 (int8)        | 3 (int8)        | Activation:none                                     |
    |       |                 | 384 (int8)        |                 |                                                     |
    |       |                 | 3 (int32)         |                 |                                                     |
    | 8     | softmax         | 3 (int8)          | 3 (int8)        | BuiltinOptionsType=9                                |
    | 9     | dequantize      | 3 (int8)          | 3 (float32)     | BuiltinOptionsType=0                                |
    +-------+-----------------+-------------------+-----------------+-----------------------------------------------------+
    Total MACs: 1.197 M
    Total OPs: 2.561 M
    Name: image_example1
    Version: 1
    Description: Image classifier example for detecting Rock/Paper/Scissors hand gestures in images
    Classes: rock, paper, scissor
    hash: 31bdc72ea90bfbcfcbe0fccaed749686
    date: 2022-04-28T17:33:35.474Z
    runtime_memory_size: 71408
    samplewise_norm.rescale: 0.0
    samplewise_norm.mean_and_std: True
    .tflite file size: 15.0kB


Model Profiling Report
-----------------------

.. code-block:: shell
   
   # Profile on physical EFR32xG24 using MVP accelerator
   mltk profile image_example1 --device --accelerator MVP

    Profiling Summary
    Name: image_example1
    Accelerator: MVP
    Input Shape: 1x96x96x1
    Input Data Type: float32
    Output Shape: 1x3
    Output Data Type: float32
    Flash, Model File Size (bytes): 15.0k
    RAM, Runtime Memory Size (bytes): 85.4k
    Operation Count: 2.7M
    Multiply-Accumulate Count: 1.2M
    Layer Count: 10
    Unsupported Layer Count: 0
    Accelerator Cycle Count: 1.5M
    CPU Cycle Count: 970.8k
    CPU Utilization (%): 40.6
    Clock Rate (hz): 78.0M
    Time (s): 30.7m
    Ops/s: 87.3M
    MACs/s: 39.0M
    Inference/s: 32.6

    Model Layers
    +-------+-----------------+--------+--------+------------+------------+----------+-------------------------+--------------+-----------------------------------------------------+
    | Index | OpCode          | # Ops  | # MACs | Acc Cycles | CPU Cycles | Time (s) | Input Shape             | Output Shape | Options                                             |
    +-------+-----------------+--------+--------+------------+------------+----------+-------------------------+--------------+-----------------------------------------------------+
    | 0     | quantize        | 36.9k  | 0      | 0          | 314.5k     | 4.0m     | 1x96x96x1               | 1x96x96x1    | Type=none                                           |
    | 1     | conv_2d         | 1.2M   | 497.7k | 902.9k     | 16.1k      | 11.4m    | 1x96x96x1,24x3x3x1,24   | 1x48x48x24   | Padding:same stride:2x2 activation:relu             |
    | 2     | average_pool_2d | 69.1k  | 0      | 48.5k      | 569.2k     | 7.6m     | 1x48x48x24              | 1x24x24x24   | Padding:valid stride:2x2 filter:2x2 activation:none |
    | 3     | conv_2d         | 842.2k | 418.2k | 326.6k     | 8.0k       | 4.2m     | 1x24x24x24,16x3x3x24,16 | 1x11x11x16   | Padding:valid stride:2x2 activation:relu            |
    | 4     | conv_2d         | 565.7k | 279.9k | 217.1k     | 10.1k      | 2.8m     | 1x11x11x16,24x3x3x16,24 | 1x9x9x24     | Padding:valid stride:1x1 activation:relu            |
    | 5     | average_pool_2d | 1.9k   | 0      | 1.6k       | 43.9k      | 540.0u   | 1x9x9x24                | 1x4x4x24     | Padding:valid stride:2x2 filter:2x2 activation:none |
    | 6     | reshape         | 0      | 0      | 0          | 2.4k       | 30.0u    | 1x4x4x24,2              | 1x384        | Type=none                                           |
    | 7     | fully_connected | 2.3k   | 1.1k   | 1.8k       | 2.2k       | 30.0u    | 1x384,3x384,3           | 1x3          | Activation:none                                     |
    | 8     | softmax         | 15.0   | 0      | 0          | 3.5k       | 60.0u    | 1x3                     | 1x3          | Type=softmaxoptions                                 |
    | 9     | dequantize      | 6.0    | 0      | 0          | 1.1k       | 0        | 1x3                     | 1x3          | Type=none                                           |
    +-------+-----------------+--------+--------+------------+------------+----------+-------------------------+--------------+-----------------------------------------------------+


Model Diagram
------------------

.. code-block:: shell
   
   mltk view image_example1 --tflite

.. raw:: html

    <div class="model-diagram">
        <a href="../../../../_images/models/image_example1.tflite.png" target="_blank">
            <img src="../../../../_images/models/image_example1.tflite.png" />
            <p>Click to enlarge</p>
        </a>
    </div>


"""

# Bring in the required Keras classes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, 
    Activation, 
    Flatten, 
    BatchNormalization,
    Conv2D,
    AveragePooling2D
)

import mltk.core as mltk_core
# By default, we use the ParallelImageDataGenerator
# We could use the Keras ImageDataGenerator but it is slower
from mltk.core.preprocess.image.parallel_generator import ParallelImageDataGenerator
#from keras.preprocessing.image import ImageDataGenerator
# Import the dataset
from mltk.datasets.image import rock_paper_scissors_v1



# Instantiate the MltkModel object with the following 'mixins':
# - TrainMixin            - Provides classifier model training operations and settings
# - ImageDatasetMixin     - Provides image data generation operations and settings
# - EvaluateClassifierMixin         - Provides classifier evaluation operations and settings
# @mltk_model # NOTE: This tag is required for this model be discoverable
class MyModel(
    mltk_core.MltkModel, 
    mltk_core.TrainMixin, 
    mltk_core.ImageDatasetMixin, 
    mltk_core.EvaluateClassifierMixin
):
    pass
my_model = MyModel()


#################################################
# General Settings
# 
my_model.version = 1
my_model.description = 'Image classifier example for detecting Rock/Paper/Scissors hand gestures in images'


#################################################
# Training Settings
my_model.epochs = 100
my_model.batch_size = 32
my_model.optimizer = 'adam'
my_model.metrics = ['accuracy']
my_model.loss = 'categorical_crossentropy'

#################################################
# Training callback Settings

# Generate a training weights .h5 whenever the 
# val_accuracy improves
my_model.checkpoint['monitor'] =  'val_accuracy'


# https://keras.io/api/callbacks/reduce_lr_on_plateau/
# If the test loss doesn't improve after 'patience' epochs 
# then decrease the learning rate by 'factor'
my_model.reduce_lr_on_plateau = dict(
  monitor='loss',
  factor = 0.95,
  min_delta=0.0001,
  patience = 1
)

# If the validation accuracy doesn't improve after 35 epochs then stop training
# https://keras.io/api/callbacks/early_stopping/
my_model.early_stopping = dict( 
  monitor = 'val_accuracy',
  patience = 35
)


#################################################
# TF-Lite converter settings
my_model.tflite_converter['optimizations'] = ['DEFAULT']
my_model.tflite_converter['supported_ops'] = ['TFLITE_BUILTINS_INT8']
my_model.tflite_converter['inference_input_type'] = 'float32' # Need to use float32 used using samplewise_center=True and samplewise_std_normalization=True
my_model.tflite_converter['inference_output_type'] = 'float32'
 # generate a representative dataset from the validation data
my_model.tflite_converter['representative_dataset'] = 'generate'



#################################################
# Image Dataset Settings

# The directory of the training data
# NOTE: This can also be a directory path
my_model.dataset = rock_paper_scissors_v1
# The classification type
my_model.class_mode = 'categorical'
# The class labels found in your training dataset directory
my_model.classes = rock_paper_scissors_v1.CLASSES
# The input shape to the model. The dataset samples will be resized if necessary
my_model.input_shape = rock_paper_scissors_v1.INPUT_SHAPE
# Shuffle the dataset directory once
my_model.shuffle_dataset_enabled = True

# The numbers of samples for each class is different
# Then ensures each class contributes equally to training the model
my_model.class_weights = 'balanced'



#################################################
# ParallelImageDataGenerator Settings

my_model.datagen = ParallelImageDataGenerator(
    cores=0.3,
    debug=False,
    max_batches_pending=8, 
    validation_split= 0.1,
    validation_augmentation_enabled=False,
    rotation_range=35,
    width_shift_range=5,
    height_shift_range=5,
    brightness_range=(0.50, 1.70),
    contrast_range=(0.50, 1.70),
    noise=['gauss', 'poisson', 's&p'],
    # zoom_range=(0.95, 1.05),
    samplewise_center=True,
    samplewise_std_normalization=True,
    rescale=None,
    horizontal_flip=True,
    vertical_flip=True,
)


#################################################
# Build the ML Model
def my_model_builder(model: MyModel):
    keras_model = Sequential(name=my_model.name)

    keras_model.add(Conv2D(24, strides=(2,2), 
                            kernel_size=3, use_bias=True, padding='same', 
                            activation='relu', input_shape=model.input_shape))
    keras_model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    keras_model.add(Conv2D(16, strides=(2,2), kernel_size=3, use_bias=True, padding='valid', activation='relu'))
    keras_model.add(Conv2D(24, strides=(1,1), kernel_size=3, use_bias=True, padding='valid'))
    keras_model.add(BatchNormalization())
    keras_model.add(Activation('relu'))
    keras_model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    keras_model.add(Flatten())
    keras_model.add(Dense(model.n_classes, use_bias=True))
    keras_model.add(Activation('softmax'))

    keras_model.compile(
        loss=model.loss, 
        optimizer=model.optimizer, 
        metrics=model.metrics
    )

    return keras_model

my_model.build_model_function = my_model_builder




# Register the "datagen_dump" custom command
import typer
@my_model.cli.command('datagen_dump')
def datagen_dump_custom_command(
    count:int = typer.Option(100, '--count',
        help='Number of samples to dump'
    ),
):
    """Custom command to dump the augmented samples
    
    \b
    Invoke this command with:
    mltk custom image_example1 datagen_dump --count 20
    """

    my_model.datagen.save_to_dir = my_model.create_log_dir('datagen_dump', delete_existing=True)
    my_model.datagen.debug = True
    my_model.datagen.cores = 1
    my_model.datagen.max_batches_pending = 1
    my_model.datagen.batch_size = 1

    my_model.load_dataset(subset='training')

    for i, _ in enumerate(my_model.x):
        if i >= count:
            break
    
    my_model.unload_dataset()

    print(f'Generated data dump to: {my_model.datagen.save_to_dir}')





##########################################################################################
# The following allows for running this model training script directly, e.g.: 
# python image_example1.py
#
# Note that this has the same functionality as:
# mltk train image_example1
#
if __name__ == '__main__':
    from mltk import cli

    # Setup the CLI logger
    cli.get_logger(verbose=False)

    # If this is true then this will do a "dry run" of the model testing
    # If this is false, then the model will be fully trained
    test_mode_enabled = True

    # Train the model
    # This does the same as issuing the command: mltk train image_example1-test --clean
    train_results = mltk_core.train_model(my_model, clean=True, test=test_mode_enabled)
    print(train_results)

    # Evaluate the model against the quantized .h5 (i.e. float32) model
    # This does the same as issuing the command: mltk evaluate image_example1-test
    tflite_eval_results = mltk_core.evaluate_model(my_model, verbose=True, test=test_mode_enabled)
    print(tflite_eval_results)

    # Profile the model in the simulator
    # This does the same as issuing the command: mltk profile image_example1-test
    profiling_results = mltk_core.profile_model(my_model, test=test_mode_enabled)
    print(profiling_results)