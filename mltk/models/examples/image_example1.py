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

.. code-block:: console

   # Do a "dry run" test training of the model
   > mltk train image_example1-test

   # Train the model
   > mltk train image_example1

   # Evaluate the trained model .tflite model
   > mltk evaluate image_example1 --tflite

   # Profile the model in the MVP hardware accelerator simulator
   > mltk profile image_example1 --accelerator MVP

   # Profile the model on a physical development board
   > mltk profile image_example1 --accelerator MVP --device



Model Summary
--------------

.. code-block:: console
    
    > mltk summarize image_example1 --tflite
    
    +-------+-----------------+-----------------+-----------------+-----------------------------------------------------+  
    | Index | OpCode          | Input(s)        | Output(s)       | Config                                              |  
    +-------+-----------------+-----------------+-----------------+-----------------------------------------------------+  
    | 0     | conv_2d         | 96x96x1 (int8)  | 48x48x24 (int8) | Padding:same stride:2x2 activation:relu             |  
    |       |                 | 3x3x1 (int8)    |                 |                                                     |  
    |       |                 | 24 (int32)      |                 |                                                     |  
    | 1     | average_pool_2d | 48x48x24 (int8) | 24x24x24 (int8) | Padding:valid stride:2x2 filter:2x2 activation:none |  
    | 2     | conv_2d         | 24x24x24 (int8) | 11x11x16 (int8) | Padding:valid stride:2x2 activation:relu            |  
    |       |                 | 3x3x24 (int8)   |                 |                                                     |  
    |       |                 | 16 (int32)      |                 |                                                     |  
    | 3     | conv_2d         | 11x11x16 (int8) | 9x9x24 (int8)   | Padding:valid stride:1x1 activation:relu            |  
    |       |                 | 3x3x16 (int8)   |                 |                                                     |  
    |       |                 | 24 (int32)      |                 |                                                     |  
    | 4     | average_pool_2d | 9x9x24 (int8)   | 4x4x24 (int8)   | Padding:valid stride:2x2 filter:2x2 activation:none |  
    | 5     | reshape         | 4x4x24 (int8)   | 384 (int8)      | BuiltinOptionsType=0                                |  
    |       |                 | 2 (int32)       |                 |                                                     |  
    | 6     | fully_connected | 384 (int8)      | 3 (int8)        | Activation:none                                     |  
    |       |                 | 384 (int8)      |                 |                                                     |  
    |       |                 | 3 (int32)       |                 |                                                     |  
    | 7     | softmax         | 3 (int8)        | 3 (int8)        | BuiltinOptionsType=9                                |  
    +-------+-----------------+-----------------+-----------------+-----------------------------------------------------+  
    Total MACs: 1.197 M                                                                                                    
    Total OPs: 2.524 M                                                                                                     
    Name: image_example1                                                                                                   
    Version: 1                                                                                                             
    Description: Image classifier example for detecting Rock/Paper/Scissors hand gestures in images                        
    Classes: rock, paper, scissor                                                                                          
    hash: b60ad56115089679c5006f04e4c9a9b0                                                                                 
    date: 2022-02-04T18:48:51.997Z                                                                                         
    runtime_memory_size: 71172                                                                                             
    samplewise_norm.rescale: 0.0                                                                                           
    samplewise_norm.mean_and_std: True                                                                                     
    .tflite file size: 15.7kB


Model Diagram
------------------

.. code-block:: console
   
   > mltk view image_example1 --tflite

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
my_model.epochs = -1 # We use the EarlyStopping keras callback to stop the training
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
  factor = 0.25,
  patience = 10
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
my_model.tflite_converter['inference_input_type'] = 'int8' # can also be float32
my_model.tflite_converter['inference_output_type'] = 'int8'
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