
# Bring in the required Keras classes
import functools
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Dense, 
    Activation, 
    Flatten, 
    BatchNormalization,
    Conv2D,
    AveragePooling2D
)
from mltk.core.keras import ImageDataGenerator
from mltk.datasets.image import cifar10
from mltk.core.model import (
    MltkModel,
    TrainMixin,
    ImageDatasetMixin,
    EvaluateClassifierMixin
)



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
mltk_model = MyModel()


#################################################
# General Settings
# 
mltk_model.version = 1
mltk_model.description = 'Model used for unit tests'


#################################################
# Training Settings
mltk_model.epochs = 5
mltk_model.batch_size = 3
mltk_model.optimizer = 'adam'
mltk_model.metrics = ['accuracy']
mltk_model.loss = 'categorical_crossentropy'

#################################################
# Training callback Settings

# Generate a training weights .h5 whenever the 
# val_accuracy improves
mltk_model.checkpoint['monitor'] =  'val_accuracy'


#################################################
# TF-Lite converter settings
mltk_model.tflite_converter['optimizations'] = ['DEFAULT']
mltk_model.tflite_converter['supported_ops'] = ['TFLITE_BUILTINS_INT8']
mltk_model.tflite_converter['inference_input_type'] = 'int8' # can also be float32
mltk_model.tflite_converter['inference_output_type'] = 'int8'
 # generate a representative dataset from the validation data
mltk_model.tflite_converter['representative_dataset'] = 'generate'



#################################################
# Image Dataset Settings

# The directory of the training data
def my_dataset_loader(model):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # Convert for training
    x_train = x_train.astype('float32')[:64]
    x_test = x_test.astype('float32')[:64]

    y_train = y_train[:64]
    y_test = y_test[:64]

    # Scale to INT8 range (simple non-adaptive)
    x_train = (x_train-128)/128
    x_test = (x_test-128)/128

    y_train = to_categorical(y_train, len(model.classes))
    y_test = to_categorical(y_test, len(model.classes))

    return  x_train, y_train, x_test, y_test


mltk_model.dataset = functools.partial(my_dataset_loader, mltk_model)

# The classification type
mltk_model.class_mode = 'categorical'
# The class labels found in your training dataset directory
mltk_model.classes =  ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# The input shape to the model. The dataset samples will be resized if necessary
mltk_model.input_shape = (32, 32, 3)



#################################################
# ImageDataGenerator Settings

mltk_model.datagen = ImageDataGenerator(
    validation_split= 0.1,
)


#################################################
# Build the ML Model
def my_model_builder(model: MyModel):
    keras_model = Sequential(name=mltk_model.name)

    keras_model.add(Conv2D(8, strides=(2,2), 
                            kernel_size=3, use_bias=True, padding='same', 
                            activation='relu', input_shape=model.input_shape))
    keras_model.add(Flatten())
    keras_model.add(Dense(model.n_classes, use_bias=True))
    keras_model.add(Activation('softmax'))

    keras_model.compile(
        loss=model.loss, 
        optimizer=model.optimizer, 
        metrics=model.metrics
    )

    return keras_model

mltk_model.build_model_function = my_model_builder