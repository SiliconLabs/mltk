"""rock_paper_scissors
*************************

- Source code: `rock_paper_scissors.py <https://github.com/siliconlabs/mltk/blob/master/mltk/models/siliconlabs/rock_paper_scissors.py>`_.
- Pre-trained model: `rock_paper_scissors.mltk.zip <https://github.com/siliconlabs/mltk/blob/master/mltk/models/siliconlabs/rock_paper_scissors.mltk.zip>`_.

This provides an example of how to define a classification model 
that uses the Rock/Paper/Scissors dataset with the ParallelImageGenerator as its data source.

The basic flow for the ML model is:

``96x96x1 grayscale image of hand gesture -> ML Model -> [result vector]``


Where `[result vector]` is a 3 element array with each element containing the % probability that the 
given image is a "rock", "paper", or "scissor" hand gesture.



Commands
--------------

.. code-block:: console

   # Do a "dry run" test training of the model
   > mltk train rock_paper_scissors-test

   # Train the model
   > mltk train rock_paper_scissors

   # Evaluate the trained model .tflite model
   > mltk evaluate rock_paper_scissors --tflite

   # Profile the model in the MVP hardware accelerator simulator
   > mltk profile rock_paper_scissors --accelerator MVP

   # Profile the model on a physical development board
   > mltk profile rock_paper_scissors --accelerator MVP --device


Model Summary
--------------

.. code-block:: console
    
    > mltk summarize rock_paper_scissors --tflite
    
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
    | 5     | reshape         | 9x9x24 (int8)     | 1944 (int8)     | BuiltinOptionsType=0                                |      
    |       |                 | 2 (int32)         |                 |                                                     |      
    | 6     | fully_connected | 1944 (int8)       | 32 (int8)       | Activation:none                                     |      
    |       |                 | 1944 (int8)       |                 |                                                     |      
    |       |                 | 32 (int32)        |                 |                                                     |      
    | 7     | fully_connected | 32 (int8)         | 3 (int8)        | Activation:none                                     |      
    |       |                 | 32 (int8)         |                 |                                                     |      
    |       |                 | 3 (int32)         |                 |                                                     |      
    | 8     | softmax         | 3 (int8)          | 3 (int8)        | BuiltinOptionsType=9                                |      
    | 9     | dequantize      | 3 (int8)          | 3 (float32)     | BuiltinOptionsType=0                                |      
    +-------+-----------------+-------------------+-----------------+-----------------------------------------------------+      
    Total MACs: 1.258 M                                                                                                          
    Total OPs: 2.683 M                                                                                                           
    Name: rock_paper_scissors                                                                                                    
    Version: 1                                                                                                                   
    Description: Image classifier example for detecting Rock/Paper/Scissors hand gestures in images                              
    Classes: rock, paper, scissor                                                                                                
    hash: b396facc6e20dabe3d68dd78685ca0c1                                                                                       
    date: 2022-02-04T19:09:54.944Z                                                                                               
    runtime_memory_size: 71356                                                                                                   
    samplewise_norm.rescale: 0.0                                                                                                 
    samplewise_norm.mean_and_std: True                                                                                           
    .tflite file size: 77.6kB


Model Diagram
------------------

.. code-block:: console
   
   > mltk view  rock_paper_scissors --tflite

.. raw:: html

    <div class="model-diagram">
        <a href="../../../../_images/models/ rock_paper_scissors.tflite.png" target="_blank">
            <img src="../../../../_images/models/ rock_paper_scissors.tflite.png" />
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

from mltk.core.model import (
    MltkModel,
    TrainMixin,
    ImageDatasetMixin,
    EvaluateClassifierMixin
)

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
    MltkModel, 
    TrainMixin, 
    ImageDatasetMixin, 
    EvaluateClassifierMixin
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
my_model.epochs = 150
my_model.batch_size = 32
my_model.optimizer = 'adam'
my_model.metrics = ['accuracy']
my_model.loss = 'categorical_crossentropy'

#################################################
# Training callback Settings

# Generate a training weights .h5 whenever the 
# val_accuracy improves
my_model.checkpoint['monitor'] =  'val_accuracy'


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



#################################################
# TF-Lite converter settings
my_model.tflite_converter['optimizations'] = ['DEFAULT']
my_model.tflite_converter['supported_ops'] = ['TFLITE_BUILTINS_INT8']
my_model.tflite_converter['inference_input_type'] = 'float32' # can also be float32
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
    cores=0.65,
    debug=False,
    max_batches_pending=32, 
    validation_split= 0.1,
    validation_augmentation_enabled=False,
    # rotation_range=5,
    width_shift_range=5,
    height_shift_range=5,
    brightness_range=(0.50, 1.70),
    contrast_range=(0.50, 1.70),
    #noise=['gauss', 'poisson', 's&p'],
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
    #keras_model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    keras_model.add(Flatten())
    keras_model.add(Dense(32, use_bias=True))
    keras_model.add(Dense(model.n_classes, use_bias=True))
    keras_model.add(Activation('softmax'))

    keras_model.compile(
        loss=model.loss, 
        optimizer=model.optimizer, 
        metrics=model.metrics
    )

    return keras_model

my_model.build_model_function = my_model_builder