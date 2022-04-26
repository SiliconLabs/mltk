
import functools
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from mltk.core.model import (
    MltkModel,
    TrainMixin,
    ImageDatasetMixin,
    EvaluateAutoEncoderMixin
)
from mltk.models.shared import FullyConnectedAutoEncoder


# Instantiate the MltkModel object with the following 'mixins':
# - TrainMixin            - Provides classifier model training operations and settings
# - ImageDatasetMixin     - Provides image data generation operations and settings
# - EvaluateAutoEncoderMixin  - Provides auto-encoder evaluation operations and settings
# @my_model   # NOTE: This tag is required for this model be discoverable
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
my_model.description = 'Testing autoencoder model'


#################################################
# Training parameters
my_model.epochs = 5
my_model.batch_size = 3 
my_model.optimizer = 'adam'
my_model.loss = 'mean_squared_error'
my_model.metrics = ['mean_squared_error']


#################################################
# Image Dataset Settings

def my_dataset_loader(model: MyModel):
    n_samples = 64
    shape = (n_samples,) + model.input_shape
    x_train = np.random.uniform(low=-1, high=1, size=shape).astype(np.float32)
    x_test = np.random.uniform(low=-1, high=1, size=shape).astype(np.float32)
    return  x_train, x_train, x_test, x_test

my_model.dataset = functools.partial(my_dataset_loader, my_model)

# Auto-Encoder directly pass the image into model
my_model.class_mode = 'input'
# The class labels found in your training dataset directory
my_model.classes = ['normal']
# Don't use weights for auto-encoders
my_model.class_weights = None
# The input shape to the model. The dataset samples will be resized if necessary
# This is the shape defined by the ToyADMOS dataset
my_model.input_shape = (5,128,1)

validation_split = 0.1


##############################################################
# Training callbacks
#

my_model.checkpoint['monitor'] = 'val_loss'
my_model.checkpoint['mode'] = 'auto'


my_model.datagen = ImageDataGenerator(
    validation_split=validation_split,
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