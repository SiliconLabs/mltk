"""image_tf_dataset
**********************

Image classification example using the Tensorflow dataset API

- Source code: `image_tf_dataset.py <https://github.com/siliconlabs/mltk/blob/master/mltk/models/examples/image_tf_dataset.py>`_

This provides an example of how to use the `Tensorflow Dataset API <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>`_
with the various Tensorflow `image augmentations <https://www.tensorflow.org/api_docs/python/tf/image>`_
to augment images during model training. 



Commands
--------------

.. code-block:: shell

   # Do a "dry run" test training of the model
   mltk train image_tf_dataset-test

   # Train the model
   mltk train image_tf_dataset

   # Evaluate the trained model .tflite model
   mltk evaluate image_tf_dataset --tflite

   # Profile the model in the MVP hardware accelerator simulator
   mltk profile image_tf_dataset --accelerator MVP

   # Profile the model on a physical development board
   mltk profile image_tf_dataset --accelerator MVP --device

   # Directly invoke the model script
   python image_tf_dataset.py


"""
import tensorflow as tf
import numpy as np

import mltk.core as mltk_core
from mltk.utils.archive_downloader import download_verify_extract
from mltk.core.preprocess.utils import tf_dataset as tf_dataset_utils
from mltk.core.preprocess.utils import image as image_utils



##########################################################################
# Instantiate the MltkModel instance
#

# @mltk_model
class MyModel(
    mltk_core.MltkModel,    # We must inherit the MltkModel class
    mltk_core.TrainMixin,   # We also inherit the TrainMixin since we want to train this model
    mltk_core.DatasetMixin, # We also need the DatasetMixin mixin to provide the relevant dataset properties
    mltk_core.EvaluateClassifierMixin,  # While not required, also inherit EvaluateClassifierMixin to help will generating evaluation stats for our classification model 
):
    pass
my_model = MyModel()


# General parameters
my_model.version = 1
my_model.description = 'Image classifier example using the Tensorflow dataset API with augmentations'


##########################################################################
# Training Basic Settings
#
my_model.epochs = 80
my_model.batch_size = 50



##########################################################################
# Define the model architecture
#
def my_model_builder(model: MyModel) -> tf.keras.Model:
    """Build the Keras model
    
    This is called by the MLTK just before training starts.

    Arguments:
        my_model: The MltkModel instance
    
    Returns:
        Compiled Keras model instance
    """
    keras_model = tf.keras.applications.MobileNetV2(
        input_shape=model.input_shape,
        alpha=0.15, 
        include_top=True,
        weights=None,
        classes=model.n_classes,
    )
    keras_model.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics= ['accuracy']
    )
    return keras_model

my_model.build_model_function = my_model_builder





##########################################################################
# Training callback Settings
#

# The MLTK enables the tf.keras.callbacks.ModelCheckpoint by default.
my_model.checkpoint['monitor'] =  'val_accuracy'

# https://keras.io/api/callbacks/reduce_lr_on_plateau/
# If the test loss doesn't improve after 'patience' epochs 
# then decrease the learning rate by 'factor'
my_model.reduce_lr_on_plateau = dict(
  monitor='val_loss',
  factor = 0.95,
  min_delta=0.001,
  patience = 1,
  verbose=1,
)

# If the  accuracy doesn't improve after 15 epochs then stop training
# https://keras.io/api/callbacks/early_stopping/
my_model.early_stopping = dict( 
  monitor = 'val_accuracy',
  patience = 15,
  verbose=1
)

# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard
my_model.tensorboard = dict(
    histogram_freq=0,       # frequency (in epochs) at which to compute activation and weight histograms 
                            # for the layers of the model. If set to 0, histograms won't be computed. 
                            # Validation data (or split) must be specified for histogram visualizations.
    write_graph=False,       # whether to visualize the graph in TensorBoard. The log file can become quite large when write_graph is set to True.
    write_images=False,     # whether to write model weights to visualize as image in TensorBoard.
    update_freq="batch",    # 'batch' or 'epoch' or integer. When using 'batch', writes the losses and metrics 
                            # to TensorBoard after each batch. The same applies for 'epoch'. 
                            # If using an integer, let's say 1000, the callback will write the metrics and losses 
                            # to TensorBoard every 1000 batches. Note that writing too frequently to 
                            # TensorBoard can slow down your training.
    profile_batch=(51,51),        # Profile the batch(es) to sample compute characteristics. 
                            # profile_batch must be a non-negative integer or a tuple of integers. 
                            # A pair of positive integers signify a range of batches to profile. 
                            # By default, it will profile the second batch. Set profile_batch=0 to disable profiling.
) 

# NOTE: You can also add manually add other KerasCallbacks
# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/
# Any callbacks specified here will override the built-in callbacks 
# (e.g. my_model.reduce_lr_on_plateau, my_model.early_stopping)
my_model.train_callbacks = [
    tf.keras.callbacks.TerminateOnNaN()
]


##########################################################################
# TF-Lite converter settings
# https://www.tensorflow.org/lite/performance/post_training_integer_quant#convert_using_integer-only_quantization
#
my_model.tflite_converter['optimizations'] = [tf.lite.Optimize.DEFAULT]
my_model.tflite_converter['supported_ops'] = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
my_model.tflite_converter['inference_input_type'] = np.float32
my_model.tflite_converter['inference_output_type'] = np.float32
 # generate a representative dataset from the validation data
my_model.tflite_converter['representative_dataset'] = 'generate'


##########################################################################
# Image Dataset Settings

# The input shape to the model. The dataset samples will be resized if necessary
my_model.input_shape = (96, 96, 3)


# The class labels found in your training dataset directory
my_model.classes =  ('person', 'non_person')
my_model.class_weights = 'balanced' # Ensure the classes samples a balanced during training

validation_split = 0.2

# Uncomment this to dump the augmented images samples to the log directory
#data_dump_dir = my_model.create_log_dir('dataset_dump')


##########################################################################
# Create the image augmentation pipeline
#
def image_augmentation(batch: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Augment a batch of images

    This does the following, for each image file path in the input batch:
    1. Read image file
    2. Resize the image to match the model input shape
    3. Apply random augmentations to the image sample 
    4. Standardize the image sample with: image_std = (img - mean(img)) / std(img)
    5. Dump the augmented image (if necessary)

    NOTE: This will be execute in parallel across *separate* subprocesses.

    Arguments:
        batch: Batch of image file paths
        seed: Batch of seeds to use for random number generation
            This ensures that the "random" augmentations are reproducible

    Return:
        Generated batch of augmented images
    """
    # This is a work-around needed for the tf.keras.preprocessing.image augmentations below
    enabled_numpy_behavior = tf_dataset_utils.enable_numpy_behavior()

    height, width, channels = my_model.input_shape
    batch_length = batch.shape[0]
    y_shape = (batch_length, height, width, channels)
    y_batch = np.empty(y_shape, dtype=np.float32)

    # For each image sample path in the current batch
    for i, image_path in enumerate(batch):
        new_seed = tf.random.experimental.stateless_split((seed[i], seed[i]), num=1)[0]
        np_seed = abs(new_seed[0]) % (2**32 - 1)
        np.random.seed(np_seed)

        x = image_utils.read_image_file(image_path)
        x = tf.keras.preprocessing.image.smart_resize(x, (height,width))

        x = tf.image.stateless_random_brightness(x, max_delta=.1, seed=new_seed)
        x = tf.image.stateless_random_contrast(x, 0.9, 1.1, seed=new_seed)
        x = tf.image.stateless_random_hue(x, 0.1, seed=new_seed)
        x = tf.image.stateless_random_saturation(x, 0.9, 1.1, seed=new_seed)

        #x = tf.image.stateless_random_flip_up_down(x, seed=new_seed)
        x = tf.image.stateless_random_flip_left_right(x, seed=new_seed)
        if enabled_numpy_behavior:
            x = tf.keras.preprocessing.image.random_channel_shift(x, .1, channel_axis=2)
            x = tf.keras.preprocessing.image.random_shear(x, 0.1, row_axis=0, col_axis=1, channel_axis=2)
            x = tf.keras.preprocessing.image.random_zoom(x, (0.90, 1.10), row_axis=0, col_axis=1, channel_axis=2)
            x = tf.keras.preprocessing.image.random_shift(x, .1, .1, row_axis=0, col_axis=1, channel_axis=2)
            x = tf.keras.preprocessing.image.random_rotation(x, 10, row_axis=0, col_axis=1, channel_axis=2)
        
        data_dump_dir = globals().get('data_dump_dir', None)
        if data_dump_dir:
            image_utils.write_image_file(data_dump_dir, x)

        x = tf.image.per_image_standardization(x)

        y_batch[i] = x

    return y_batch


# At the end of the augmentation pipeline we're using: x = tf.image.per_image_standardization(x)
# As such, we add a "model parameter" indicating that the image samples are normalized.
# This will be embedded into the generated .tflite. 
# At runtime, the embedded device should retrieve this parameter and normalize the images 
# before sending to the Tensorflow-Lite Micro interpreter for classification.
# See https://siliconlabs.github.io/mltk/docs/guides/model_parameters.html
my_model.model_parameters['samplewise_norm.mean_and_std'] = True


##########################################################################
# Define the MltkDataset object
# NOTE: This class is optional but is useful for organizing the code
#
class MyDataset(mltk_core.MltkDataset):

    def __init__(self):
        super().__init__()
        self.dataset_dir = ''
        self.pools = []
            
    def load_dataset(
        self, 
        subset: str,  
        test:bool = False,
        **kwargs
    ):
        """Load the dataset subset
        
        This is called automatically by the MLTK before training
        or evaluation.
        
        Args:
            subset: The dataset subset to return: 'training' or 'evaluation'
            test: This is optional, it is used when invoking a training "dryrun", e.g.: mltk train image_tf_dataset-test
                If this is true, then only return a small portion of the dataset for testing purposes

        Return:
            if subset == training:
                A tuple, (train_dataset, None, validation_dataset)
            else:
                validation_dataset
        """

        if not self.dataset_dir:
            self.dataset_dir = download_verify_extract(
                url='https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/vw_coco2014_96.tar.gz',
                dest_subdir='datasets/mscoco14/preprocessed/v1',
                file_hash='A5A465082D3F396407F8B5ABAF824DD5B28439C4',
                show_progress=True,
                remove_root_dir=True
            )
        if subset == 'training':
            x = self._load_subset('training', test=test)
            validation_data = self._load_subset('validation', test=test)

            return x, None, validation_data

        else:
            x = self._load_subset('validation', test=test)
            return x



    def _load_subset(self, subset:str, test:bool) -> tf.data.Dataset:
        if subset in ('validation', 'evaluation'):
            split = (0, validation_split)
            data_dump_dir = globals().get('data_dump_dir', None)
            if data_dump_dir:
                print(f'\n\n*** Dumping augmented samples to: {data_dump_dir}\n\n')

        else:
            split = (validation_split, 1)


        # Create a tf.data.Dataset from the extract image dataset directory
        max_samples_per_class = my_model.batch_size if test else -1
        features_ds, labels_ds = tf_dataset_utils.load_image_directory(
            directory=self.dataset_dir,
            classes=my_model.classes,
            onehot_encode=True, # We're using categorical cross-entropy so one-hot encode the labels
            shuffle=True,
            seed=42,
            max_samples_per_class=max_samples_per_class,
            split=split,
            return_image_data=False,  # We only want to return the file paths
            class_counts=my_model.class_counts[subset], 
        )


        # We use an incrementing counter as the seed for the random augmentations
        # This helps to keep the training reproducible
        seed_counter = tf.data.experimental.Counter()
        features_ds = features_ds.zip((features_ds, seed_counter))


        # Usage of tf_dataset_utils.parallel_process()
        # is optional, but can speed-up training as the data augmentations
        # are spread across the available CPU cores.
        # Each CPU core gets its own subprocess,
        # and and subprocess executes image_augmentation() on batches of the dataset.
        per_job_batch_size = my_model.batch_size * 100
        features_ds = features_ds.batch(per_job_batch_size // 100, drop_remainder=True)
        features_ds, pool = tf_dataset_utils.parallel_process(
            features_ds,
            image_augmentation,
            dtype=np.float32,
            n_jobs=.65 if subset == 'training' else .35,
            #n_jobs=64 if subset == 'training' else 28, # Configuration for 96 CPU cloud machine
            name=subset,
        )
        self.pools.append(pool)
        features_ds = features_ds.unbatch()


        # Pre-fetching batches can help with throughput
        features_ds = features_ds.prefetch(per_job_batch_size)
        
        # Combine the augmented audio samples with their corresponding labels
        ds = tf.data.Dataset.zip((features_ds, labels_ds))

        # Shuffle the data for each sample
        # A perfect shuffle would use n_samples but this can slow down training,
        # so we just shuffle batches of the data
        ds = ds.shuffle(per_job_batch_size, reshuffle_each_iteration=True, seed=42)

        # At this point we have a flat dataset of x,y tuples
        # Batch the data as necessary for training
        ds = ds.batch(my_model.batch_size)

        # Pre-fetch a couple training batches to aid throughput
        ds = ds.prefetch(2)

        return ds

    def unload_dataset(self):
        """Unload the dataset by shutting down the processing pools"""
        for pool in self.pools:
            pool.shutdown()
 
my_model.dataset = MyDataset()



##########################################################################################
# The following allows for running this model training script directly, e.g.: 
# python image_tf_dataset.py
#
# Note that this has the same functionality as:
# mltk train image_tf_dataset
#
if __name__ == '__main__':
    from mltk import cli

    # Setup the CLI logger
    cli.get_logger(verbose=False)

    # If this is true then this will do a "dry run" of the model testing
    # If this is false, then the model will be fully trained
    test_mode_enabled = True

    # Train the model
    # This does the same as issuing the command: mltk train image_tf_dataset-test --clean
    train_results = mltk_core.train_model(my_model, clean=True, test=test_mode_enabled)
    print(train_results)

    # Evaluate the model against the quantized .h5 (i.e. float32) model
    # This does the same as issuing the command: mltk evaluate image_tf_dataset-test
    tflite_eval_results = mltk_core.evaluate_model(my_model, verbose=True, test=test_mode_enabled)
    print(tflite_eval_results)

    # Profile the model in the simulator
    # This does the same as issuing the command: mltk profile image_tf_dataset-test
    profiling_results = mltk_core.profile_model(my_model, test=test_mode_enabled)
    print(profiling_results)