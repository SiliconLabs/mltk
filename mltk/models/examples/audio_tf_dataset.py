"""audio_tf_dataset
************************

Audio classification example using the Tensorflow dataset API

- Source code: `audio_tf_dataset.py <https://github.com/siliconlabs/mltk/blob/master/mltk/models/examples/audio_tf_dataset.py>`_

This provides an example of how to use the `Tensorflow Dataset API <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>`_
with the third-party Python library `audiomentations <https://github.com/iver56/audiomentations>`_
to augment audio during model training. 


Commands
--------------

.. code-block:: shell

   # Do a "dry run" test training of the model
   mltk train audio_tf_dataset-test

   # Train the model
   mltk train audio_tf_dataset

   # Evaluate the trained model .tflite model
   mltk evaluate audio_tf_dataset --tflite

   # Profile the model in the MVP hardware accelerator simulator
   mltk profile audio_tf_dataset --accelerator MVP

   # Profile the model on a physical development board
   mltk profile audio_tf_dataset --accelerator MVP --device

   # Directly invoke the model script
   python audio_tf_dataset.py


"""
# pylint: disable=redefined-outer-name
from typing import Tuple
import numpy as np
import tensorflow as tf


import mltk.core as mltk_core
from mltk.core.preprocess.audio.audio_feature_generator import AudioFeatureGeneratorSettings
from mltk.core.preprocess.utils import tf_dataset as tf_dataset_utils
from mltk.core.preprocess.utils import audio as audio_utils 
from mltk.core.preprocess.utils import image as image_utils
from mltk.datasets.audio.speech_commands import speech_commands_v2
from mltk.utils.python import install_pip_package


# Install the audiomentations Python package (if necessary) 
# then import it
install_pip_package('audiomentations')
import audiomentations



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



# General Settings
my_model.version = 1
my_model.description = 'Audio classifier example using the Tensorflow dataset API with augmentations'


##########################################################################
# Training Basic Settings
#
my_model.epochs = 100
my_model.batch_size = 32 



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

    input_shape = model.input_shape
    filters = 8 

    keras_model = tf.keras.models.Sequential(name=model.name, layers = [
        tf.keras.layers.Conv2D(filters, (3,3), padding='same', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(2*filters,(3,3), padding='same'), 
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(4*filters, (3,3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers. MaxPooling2D(2,2),
    
        tf.keras.layers.Conv2D(4*filters, (3,3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(4*filters, (3,3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(model.n_classes, activation='softmax')
    ])

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
# If the test accuracy doesn't improve after 'patience' epochs 
# then decrease the learning rate by 'factor'
my_model.reduce_lr_on_plateau = dict(
  monitor='accuracy',
  factor = 0.95,
  patience = 1,
  min_delta=0.01
)

# https://keras.io/api/callbacks/early_stopping/
# If the validation accuracy doesn't improve after 'patience' epochs then stop training
my_model.early_stopping = dict( 
  monitor = 'val_accuracy',
  patience = 15 
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
my_model.tflite_converter['inference_input_type'] = np.int8
my_model.tflite_converter['inference_output_type'] = np.int8
 # generate a representative dataset from the validation data
my_model.tflite_converter['representative_dataset'] = 'generate'



##########################################################################
# Specify AudioFeatureGenerator Settings
# See https://siliconlabs.github.io/mltk/docs/audio/audio_feature_generator.html
#
frontend_settings = AudioFeatureGeneratorSettings()

frontend_settings.sample_rate_hz = 16000
frontend_settings.sample_length_ms = 1000
frontend_settings.window_size_ms = 32
frontend_settings.window_step_ms = 16
frontend_settings.filterbank_n_channels = 64
frontend_settings.filterbank_upper_band_limit = 4000.0-1 # Spoken language usually only goes up to 4k
frontend_settings.filterbank_lower_band_limit = 100.0
frontend_settings.noise_reduction_enable = False # Disable the noise reduction block
frontend_settings.noise_reduction_smoothing_bits = 5
frontend_settings.noise_reduction_even_smoothing = 0.004
frontend_settings.noise_reduction_odd_smoothing = 0.004
frontend_settings.noise_reduction_min_signal_remaining = 0.05
frontend_settings.pcan_enable = False
frontend_settings.pcan_strength = 0.95
frontend_settings.pcan_offset = 80.0
frontend_settings.pcan_gain_bits = 21
frontend_settings.log_scale_enable = True
frontend_settings.log_scale_shift = 6

frontend_settings.activity_detection_enable = True # Enable the activity detection block
frontend_settings.activity_detection_alpha_a = 0.5
frontend_settings.activity_detection_alpha_b = 0.8
frontend_settings.activity_detection_arm_threshold = 0.75
frontend_settings.activity_detection_trip_threshold = 0.8

frontend_settings.dc_notch_filter_enable = True # Enable the DC notch filter
frontend_settings.dc_notch_filter_coefficient = 0.95

frontend_settings.quantize_dynamic_scale_enable = True # Enable dynamic quantization
frontend_settings.quantize_dynamic_scale_range_db = 40.0

# Add the Audio Feature generator settings to the model parameters
# This way, they are included in the generated .tflite model file
# See https://siliconlabs.github.io/mltk/docs/guides/model_parameters.html
my_model.model_parameters.update(frontend_settings)


##########################################################################
# Specify the other dataset settings
#

my_model.input_shape = frontend_settings.spectrogram_shape + (1,)

my_model.classes = ['left', 'right', 'up', 'down', '_unknown_']
my_model.class_weights = 'balanced' # Ensure the classes samples a balanced during training

validation_split = 0.2

# Uncomment this to dump the augmented audio samples to the log directory
#data_dump_dir = my_model.create_log_dir('dataset_dump')



##########################################################################
# Create the audio augmentation pipeline
#
def audio_augmentation_pipeline(batch:np.ndarray, seed:np.ndarray) -> np.ndarray:
    """Augment a batch of audio clips and generate spectrograms

    This does the following, for each audio file path in the input batch:
    1. Read audio file
    2. Adjust its length to fit within the specified length
    3. Apply random augmentations to the audio sample using audiomentations
    4. Convert to the specified sample rate (if necessary)
    5. Generate a spectrogram from the augmented audio sample
    6. Dump the augmented audio and spectrogram (if necessary)

    NOTE: This will be execute in parallel across *separate* subprocesses.

    Arguments:
        batch: Batch of audio file paths
        seed: Batch of seeds to use for random number generation,
            This ensures that the "random" augmentations are reproducible

    Return:
        Generated batch of spectrograms from augmented audio samples
    """
    batch_length = batch.shape[0]
    height, width = frontend_settings.spectrogram_shape
    y_shape = (batch_length, height, width, 1)
    y_batch = np.empty(y_shape, dtype=np.int8)
    
    # For each audio sample path in the current batch
    for i, audio_path in enumerate(batch):
        new_seed = tf.random.experimental.stateless_split((seed[i], seed[i]), num=1)[0][0]
        np_seed = abs(new_seed) % (2**32 - 1)
        np.random.seed(np_seed)

        # Read the audio file
        sample, original_sample_rate = audio_utils.read_audio_file(
            audio_path, 
            return_sample_rate=True,
            return_numpy=True
        )

        # Adjust the audio clip to the length defined in the frontend_settings
        out_length = int((original_sample_rate * frontend_settings.sample_length_ms) / 1000)
        adjusted_sample = audio_utils.adjust_length(
            sample,
            out_length=out_length,
            trim_threshold_db=30,
            offset=np.random.uniform(0, 1)
        )

        # Initialize the global audio augmentations instance
        # NOTE: We want this to be global so that we only initialize it once per subprocess
        audio_augmentations = globals().get('audio_augmentations', None)
        if audio_augmentations is None:
            dataset_dir = speech_commands_v2.load_data()
            audio_augmentations = audiomentations.Compose(
                p=0.90,
                transforms=[ 
                audiomentations.PitchShift(min_semitones=-1, max_semitones=1, p=0.5),
                audiomentations.TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
                audiomentations.Gain(min_gain_in_db=0.95, max_gain_in_db=2, p=0.75),
                audiomentations.AirAbsorption(
                    min_temperature = 10.0,
                    max_temperature = 20.0,
                    min_humidity = 30.0,
                    max_humidity = 90.0,
                    min_distance = 0.5,
                    max_distance = 7.0,
                    p=0.5,
                ),
                audiomentations.AddBackgroundNoise(
                    f'{dataset_dir}/_background_noise_', 
                    min_snr_in_db=20,
                    max_snr_in_db=40,
                    noise_rms="relative",
                    lru_cache_size=10,
                    p=0.75
                ),
            ])
            globals()['audio_augmentations'] = audio_augmentations

        # Apply random augmentations to the audio sample
        augmented_sample = audio_augmentations(adjusted_sample, original_sample_rate)

        # Convert the sample rate (if necessary)
        if original_sample_rate != frontend_settings.sample_rate_hz:
            augmented_sample = audio_utils.resample(
                augmented_sample, 
                orig_sr=original_sample_rate, 
                target_sr=frontend_settings.sample_rate_hz
            )

        # Generate a spectrogram from the augmented audio sample
        spectrogram = audio_utils.apply_frontend(
            sample=augmented_sample, 
            settings=frontend_settings, 
            dtype=np.int8
        )
        # The output spectrogram is 2D, add a channel dimension to make it 3D:
        # (height, width, channels=1)
        spectrogram = np.expand_dims(spectrogram, axis=-1)

        # Dump the augmented audio sample AND corresponding spectrogram (if necessary)
        data_dump_dir = globals().get('data_dump_dir', None)
        if data_dump_dir:
            audio_dump_path = audio_utils.write_audio_file(
                data_dump_dir, 
                augmented_sample, 
                sample_rate=frontend_settings.sample_rate_hz
            )
            image_dump_path = audio_dump_path.replace('.wav', '.jpg')
            image_utils.write_image_file(image_dump_path, spectrogram)
            
        y_batch[i] = spectrogram

    return y_batch


##########################################################################
# Define the MltkDataset object
# NOTE: This class is optional but is useful for organizing the code
#
class MyDataset(mltk_core.MltkDataset):

    def __init__(self):
        super().__init__()
        self.pools = []

    def load_dataset(
        self, 
        subset: str,  
        test:bool = False,
        **kwargs
    ) -> Tuple[tf.data.Dataset, None, tf.data.Dataset]:
        """Load the dataset subset
        
        This is called automatically by the MLTK before training
        or evaluation.
        
        Args:
            subset: The dataset subset to return: 'training' or 'evaluation'
            test: This is optional, it is used when invoking a training "dryrun", e.g.: mltk train audio_tf_dataset-test
                If this is true, then only return a small portion of the dataset for testing purposes

        Return:
            if subset == training:
                A tuple, (train_dataset, None, validation_dataset)
            else:
                validation_dataset
        """

        if subset == 'training':
            x = self._load_subset('training', test=test)
            validation_data = self._load_subset('validation', test=test)

            return x, None, validation_data

        else:
            x = self._load_subset('validation', test=test)
            return x

    def unload_dataset(self):
        """Unload the dataset by shutting down the processing pools"""
        for pool in self.pools:
            pool.shutdown()
        self.pools.clear()


    def _load_subset(self, subset:str, test:bool) -> tf.data.Dataset:
        """Load the subset"""
        if subset == 'validation':
            split = (0, validation_split)
        else:
            split = (validation_split, 1)
            data_dump_dir = globals().get('data_dump_dir', None)
            if data_dump_dir:
                print(f'\n\n*** Dumping augmented samples to: {data_dump_dir}\n\n')


        # Download and extract the "Speech Commands" dataset
        dataset_dir = speech_commands_v2.load_data()

        # Create a tf.data.Dataset from the extracted "Speech Commands" directory
        max_samples_per_class = my_model.batch_size if test else -1
        features_ds, labels_ds = tf_dataset_utils.load_audio_directory(
            directory=dataset_dir,
            classes=my_model.classes,
            onehot_encode=True, # We're using categorical cross-entropy so one-hot encode the labels
            shuffle=True,
            seed=42,
            max_samples_per_class=max_samples_per_class,
            split=split,
            unknown_class_percentage=2.0, # We want 2x the number of "unknown" samples
            return_audio_data=False, # We only want to return the file paths
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
        # and and subprocess executes audio_augmentation_pipeline() on batches of the dataset.
        per_job_batch_size = my_model.batch_size * 100
        features_ds = features_ds.batch(per_job_batch_size // 100, drop_remainder=True)
        features_ds, pool = tf_dataset_utils.parallel_process(
            features_ds,
            audio_augmentation_pipeline,
            dtype=np.int8,
            #n_jobs=64 if subset == 'training' else 22, # These are the settings for a 96 CPU core cloud machine
            n_jobs=.65 if subset == 'training' else .35,
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
        ds = ds.shuffle(per_job_batch_size, reshuffle_each_iteration=True)
        
        # At this point we have a flat dataset of x,y tuples
        # Batch the data as necessary for training
        ds = ds.batch(my_model.batch_size)

        # Pre-fetch a couple training batches to aid throughput
        ds = ds.prefetch(2)

        return ds

my_model.dataset = MyDataset()








#################################################
# Audio Classifier Settings
#
# These are additional parameters to include in
# the generated .tflite model file.
# The settings are used by the "classify_audio" command
# or audio_classifier example application.
# NOTE: Corresponding command-line options will override these values.

# Controls the smoothing. 
# Drop all inference results that are older than <now> minus window_duration
# Longer durations (in milliseconds) will give a higher confidence that the results are correct, but may miss some commands
my_model.model_parameters['average_window_duration_ms'] = 1000
# Minimum averaged model output threshold for a class to be considered detected, 0-255. Higher values increase precision at the cost of recall
my_model.model_parameters['detection_threshold'] = 165
# Amount of milliseconds to wait after a keyword is detected before detecting new keywords
my_model.model_parameters['suppression_ms'] = 750
# The minimum number of inference results to average when calculating the detection value
my_model.model_parameters['minimum_count'] = 3
# Set the volume gain scaler (i.e. amplitude) to apply to the microphone data. If 0 or omitted, no scaler is applied
my_model.model_parameters['volume_gain'] = 2
# This the amount of time in milliseconds an audio loop takes.
my_model.model_parameters['latency_ms'] = 100
# Enable verbose inference results
my_model.model_parameters['verbose_model_output_logs'] = False




##########################################################################################
# The following allows for running this model training script directly, e.g.: 
# python audio_tf_dataset.py
#
# Note that this has the same functionality as:
# mltk train audio_tf_dataset
#
if __name__ == '__main__':
    from mltk import cli

    # Setup the CLI logger
    cli.get_logger(verbose=False)

    # If this is true then this will do a "dry run" of the model testing
    # If this is false, then the model will be fully trained
    test_mode_enabled = True

    # Train the model
    # This does the same as issuing the command: mltk train audio_tf_dataset-test --clean
    train_results = mltk_core.train_model(my_model, clean=True, test=test_mode_enabled)
    print(train_results)

    # Evaluate the model against the quantized .h5 (i.e. float32) model
    # This does the same as issuing the command: mltk evaluate audio_tf_dataset-test
    tflite_eval_results = mltk_core.evaluate_model(my_model, verbose=True, test=test_mode_enabled)
    print(tflite_eval_results)

    # Profile the model in the simulator
    # This does the same as issuing the command: mltk profile audio_tf_dataset-test
    profiling_results = mltk_core.profile_model(my_model, test=test_mode_enabled)
    print(profiling_results)