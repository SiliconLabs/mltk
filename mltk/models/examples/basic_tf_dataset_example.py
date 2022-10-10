"""basic_tf_dataset_example
*********************************

Basic MLTK model example using the Tensorflow dataset API

- Source code: `basic_tf_dataset_example.py <https://github.com/siliconlabs/mltk/blob/master/mltk/models/examples/basic_tf_dataset_example.py>`_

This provides a basic example of how to create a `model specification <https://siliconlabs.github.io/mltk/docs/guides/model_specification.html>`_
using the `Tensorflow Dataset API <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>`_ for dataset management.


Commands
--------------

.. code-block:: shell

   # Do a "dry run" test training of the model
   mltk train basic_tf_dataset_example-test

   # Train the model
   mltk train basic_tf_dataset_example

   # Evaluate the trained model .tflite model
   mltk evaluate basic_tf_dataset_example --tflite

   # Profile the model in the MVP hardware accelerator simulator
   mltk profile basic_tf_dataset_example --accelerator MVP

   # Profile the model on a physical development board
   mltk profile basic_tf_dataset_example --accelerator MVP --device

   # Directly invoke the model script
   python basic_tf_dataset_example.py

"""

import numpy as np
import tensorflow as tf
from mltk import core as mltk_core


# For this example we use the Fashion MNIST dataset.
# NOTE: The following uses the MLTK Fashion mnist dataset,
#       but we could have also used: tf.keras.datasets.fashion_mnist
from mltk.datasets.image import fashion_mnist


##########################################################################
# Prepare the model parameters
# Model / data parameters
classes = fashion_mnist.CLASSES
num_classes = len(classes)
input_shape = (28, 28, 1)
validation_split = 0.1
batch_size = 128
epochs = 15


##########################################################################
# Prepare the dataset
def my_dataset_loader(subset:str, test:bool, **kwargs) -> tf.data.Dataset:
    """Load the dataset subset
    
    This is called automatically by the MLTK before training
    or evaluation.
    
    Args:
        subset: The dataset subset to return: 'training' or 'evaluation'
        test: This is optional, it is used when invoking a training "dryrun", e.g.: mltk train basic_tf_dataset_example-test
            If this is true, then only return a small portion of the dataset for testing purposes

    Return:
        A  tf.data.Dataset that generates tuples of batches of x,y
    """
    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # If we're just testing, then truncate the dataset
    if test:
        x_test = x_test[:batch_size*3]
        x_train = x_train[:batch_size*3]
        y_train = y_train[:batch_size*3]
        y_test = y_test[:batch_size*3]


    if subset == 'training':
        n_validation_samples = int(len(x_train)*validation_split)
        n_train_samples = len(x_train) - n_validation_samples
        x_validation = x_train[n_train_samples:]
        y_validation = y_train[n_train_samples:]
        x_train = x_train[:n_train_samples]
        y_train = y_train[:n_train_samples]

        train_dataset = _create_dataset_subset(x_train, y_train, subset='training')
        validation_dataset = _create_dataset_subset(x_validation, y_validation, subset='validation')

        return train_dataset, None, validation_dataset

    else:
        evaluation_dataset = _create_dataset_subset(x_test, y_test, subset='evaluation')
        return evaluation_dataset


def _create_dataset_subset(x:np.ndarray, y: np.ndarray, subset:str) -> tf.data.Dataset:
    """Create a Tensorflow Dataset for the given x,y subset"""
    features_ds = tf.data.Dataset.from_tensor_slices(x)
    labels_ds = tf.data.Dataset.from_tensor_slices(y)

    # We're using categorical cross-entropy, 
    # so we need the labels to be one-hot encoded
    labels_ds = labels_ds.map(
        lambda label: tf.one_hot(label, depth=num_classes, dtype=tf.int32),
    )

    dataset = tf.data.Dataset.zip((features_ds, labels_ds))
    dataset = dataset.shuffle(len(x), seed=42)
    dataset = dataset.batch(batch_size)

    # This is optional, but useful for automatically generating a summary of the dataset
    my_model.class_counts[subset] = { my_model.classes[class_id]: count for (class_id, count) in enumerate(np.bincount(y)) }

    return dataset



##########################################################################
# Build the model
def my_model_builder(my_model: mltk_core.MltkModel) -> tf.keras.Model:
    """Build the Keras model
    
    This is called by the MLTK just before training starts.

    Arguments:
        my_model: The MltkModel instance
    
    Returns:
        Compiled Keras model instance
    """
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        loss="categorical_crossentropy", 
        optimizer="adam", 
        metrics=["accuracy"]
    )

    return model


##########################################################################
# Create the MltkModel instance
# and set the various properties

# @mltk_model
class MyModel(
    mltk_core.MltkModel,    # We must inherit the MltkModel class
    mltk_core.TrainMixin,   # We also inherit the TrainMixin since we want to train this model
    mltk_core.DatasetMixin, # We also need the DatasetMixin mixin to provide the relevant dataset properties
    mltk_core.EvaluateClassifierMixin,  # While not required, also inherit EvaluateClassifierMixin to help will generating evaluation for our classification model 
):
    pass

my_model = MyModel()

# These properties are optional 
# but a useful for tracking the generated .tflite
my_model.version = 1
my_model.description = 'Basic Tensorflow Dataset API example'
my_model.classes = classes
my_model.class_weights = 'balanced' # Automatically generate balanced class weights for training

# Required: Set the model build function
my_model.build_model_function = my_model_builder


# Set the other model properties
# The values are passed directly to the model.fit() API
# https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
my_model.batch_size = batch_size
my_model.epochs = epochs

# NOTE: All the other fit() arguments may also be set in the model, e.g.:
# my_model.x = my_tf_training_dataset
# my_model.validation_data = my_tf_validation_dataset
# my_model.step_per_epoch = 60

# Set the dataset
my_model.dataset = my_dataset_loader
# NOTE: Since my_dataset_loader() returns the TF dataset
# We do not need to manually set the my_model.x, my_model.validation_data properties.

# NOTE: You can also add the various KerasCallbacks
# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/
my_model.train_callbacks = [
    tf.keras.callbacks.TerminateOnNaN()
]


##########################################################################
# Specify the .tflite conversion parameters
# This is used to convert the float32 model to int8 model 
# that can run on the embedded device.


def my_representative_dataset_generator():
    """A representative dataset is required generate the .tflite
    In this example we just take the first 100 samples from the validation set.

    For more details, see:
    https://www.tensorflow.org/lite/performance/post_training_integer_quant#convert_using_integer-only_quantization

    NOTE: Quantization is automatically done at the end of training.
    It may also be invoked with:
    mltk train basic_example
    """
    for x, y in my_model.x.unbatch().batch(1).take(100):
        yield [x]

my_model.tflite_converter['inference_input_type'] = tf.float32
my_model.tflite_converter['inference_output_type'] = tf.float32
my_model.tflite_converter['representative_dataset'] = my_representative_dataset_generator


##########################################################################################
# (Optional) Configure model parameters
#
# While not required, user-defined parameters may be embedded into the .tflite model file.
# These parameters may then be read by the embedded device at runtime.
# 
# This is useful for syncing data preprocessing parameters between the model training
# script and embedded device.

# In my_dataset_loader() we scaled the image data by 1/255.
# This same scaling must also happen on the embedded device.
# Here, we're embedding the scaling value as "metadata" into the generated .tflite.
# At runtime, the embedded device should read this value from the .tflite
# and use it accordingly.
my_model.model_parameters['samplewise_norm.rescale'] = 1/255.

# Most standard Python data types may be embedded
# See: https://siliconlabs.github.io/mltk/docs/guides/model_parameters.html 
my_model.model_parameters['my_boolean'] = True 
my_model.model_parameters['my_string'] = 'This string will be embedded into the .tflite' 
my_model.model_parameters['my_bytes'] = b'This byte string will be embedded also'
my_model.model_parameters['my_float_list'] = [4.5, 2., 3.14]





##########################################################################################
# The following allows for running this model training script directly, e.g.: 
# python basic_tf_dataset_example.py
#
# Note that this has the similar functionality to:
# mltk train basic_tf_dataset_example
#
if __name__ == '__main__':
    from mltk import cli

    # Setup the CLI logger
    cli.get_logger(verbose=False)

    # If this is true then this will do a "dry run" of the model testing
    # If this is false, then the model will be fully trained
    test_mode_enabled = True

    # Train the model
    # This does the same as issuing the command: mltk train basic_tf_dataset_example-test --clean
    train_results = mltk_core.train_model(my_model, clean=True, test=test_mode_enabled)
    print(train_results)

    # Evaluate the model against the quantized .h5 (i.e. float32) model
    # This does the same as issuing the command: mltk evaluate basic_tf_dataset_example-test
    tflite_eval_results = mltk_core.evaluate_model(my_model, verbose=True, test=test_mode_enabled)
    print(tflite_eval_results)

    # Profile the model in the simulator
    # This does the same as issuing the command: mltk profile basic_tf_dataset_example-test
    profiling_results = mltk_core.profile_model(my_model, test=test_mode_enabled)
    print(profiling_results)