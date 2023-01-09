"""kws_bc_resnet
************************

"""
# pylint: disable=redefined-outer-name
from typing import Tuple
import numpy as np
import logging
import tensorflow as tf
from mltk.utils.python import install_pip_package

# Install the audiomentations Python package (if necessary) 
# then import it
install_pip_package('audiomentations')
install_pip_package('tensorflow-model-optimization', 'tensorflow_model_optimization')
import audiomentations

import mltk.core as mltk_core
from mltk.core.preprocess.audio.audio_feature_generator import AudioFeatureGeneratorSettings
from mltk.core.preprocess.utils import tf_dataset as tf_dataset_utils
from mltk.core.preprocess.utils import audio as audio_utils 
from mltk.core.preprocess.utils import image as image_utils
from mltk.datasets.audio.speech_commands import speech_commands_v2
from mltk.utils.python import DictObject
from mltk.models.shared.kws_streaming.models import bc_resnet
logger = mltk_core.get_mltk_logger()




flags = DictObject( 
    data_url='', 
    data_dir='./data2', 
    lr_schedule='linear', 
    optimizer='adam', 
    background_volume=0.1, 
    l2_weight_decay=0.0, 
    background_frequency=0.8, 
    split_data=1, 
    silence_percentage=10.0,
    unknown_percentage=10.0, 
    time_shift_ms=100.0, 
    sp_time_shift_ms=0.0, 
    testing_percentage=10, 
    validation_percentage=10, 
    how_many_training_steps='100,100,100,100,30000,30000,20000,10000,5000,5000', 
    eval_step_interval=400, 
    learning_rate='0.001,0.002,0.003,0.004,0.005,0.002,0.0005,1e-5,1e-6,1e-7', 
    batch_size=100, 
    wanted_words='yes,no,up,down,left,right,on,off,stop,go', 
    train_dir='./models_data_v2_12_labels/bc_resnet_1', 
    save_step_interval=100, 
    start_checkpoint='', 
    verbosity=0, 
    optimizer_epsilon=1e-08,
    resample=0.1, 
    sp_resample=0.0, 
    volume_resample=0.0, 
    train=1,
    sample_rate=16000, 
    clip_duration_ms=1000, 
    window_size_ms=30.0, 
    window_stride_ms=10.0, 
    preprocess='raw', 
    feature_type='mfcc_tf', 
    preemph=0.0, 
    window_type='hann', 
    mel_lower_edge_hertz=125.0, 
    mel_upper_edge_hertz=7500.0, 
    micro_enable_pcan=1, 
    micro_features_scale=0.0390625, 
    micro_min_signal_remaining=0.05, 
    micro_out_scale=1, 
    log_epsilon=1e-12, 
    dct_num_features=0, 
    use_tf_fft=0, 
    mel_non_zero_only=1,
    fft_magnitude_squared=False, 
    mel_num_bins=40, 
    use_spec_augment=0, 
    time_masks_number=2, 
    time_mask_max_size=25, 
    frequency_masks_number=2, 
    frequency_mask_max_size=7, 
    use_spec_cutout=0, 
    spec_cutout_masks_number=3, 
    spec_cutout_time_mask_size=10, 
    spec_cutout_frequency_mask_size=5, 
    return_softmax=0, novograd_beta_1=0.95, 
    novograd_beta_2=0.5, 
    novograd_weight_decay=0.001, 
    novograd_grad_averaging=0, 
    pick_deterministically=1, 
    causal_data_frame_padding=0, 
    wav=1, 
    quantize=0, 
    use_quantize_nbit=0, 
    nbit_activation_bits=8, 
    nbit_weight_bits=8, 
    data_stride=1, 
    restore_checkpoint=0, 
    model_name='bc_resnet', 
    dropouts='0.1, 0.1, 0.1, 0.1', 
    filters='8, 12, 16, 20', 
    blocks_n='2, 2, 4, 4', 
    strides='(1,1),(1,2),(1,2),(1,1)', 
    dilations='(1,1),(2,1),(4,1),(8,1)', 
    paddings='same', 
    first_filters=16, last_filters=32, 
    sub_groups=5, 
    pools='1, 1, 1, 1', 
    max_pool=0, 
    label_count=12, 
    desired_samples=16000, 
    window_size_samples=480, 
    window_stride_samples=160, 
    spectrogram_length=98, 
    data_frame_padding=None, 
    summaries_dir='./models_data_v2_12_labels/bc_resnet_1\\logs/', 
    training=True
)



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
my_model.epochs = 9999 # Training will stop based on the custom LearnRateScheduleCallback below
my_model.batch_size = 100 



##########################################################################
# Training callback Settings
#


# The MLTK enables the tf.keras.callbacks.ModelCheckpoint by default.
my_model.checkpoint['monitor'] =  'val_accuracy'

# https://keras.io/api/callbacks/reduce_lr_on_plateau/
# If the test accuracy doesn't improve after 'patience' epochs 
# then decrease the learning rate by 'factor'
# my_model.reduce_lr_on_plateau = dict(
#   monitor='accuracy',
#   factor = 0.90,
#   patience = 3,
#   min_delta=0.001,
#   cooldown=5,
# )


class LearnRateScheduleBatchCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.step = 0
        self.prev_lr = 0
        how_many_training_steps = [int(x.strip()) for x in flags['how_many_training_steps'].split(',')]
        learning_rate_steps = [float(x.strip()) for x in flags['learning_rate'].split(',')]
        self.lr_schedule = [] 
        self.printed_schedule = False
        steps_sum = 0
        for step, lr in zip(how_many_training_steps, learning_rate_steps):
            steps_sum += step
            self.lr_schedule.append((steps_sum, lr))


    def on_train_batch_begin(self, batch, logs=None):
        super().on_train_batch_begin(batch=batch, logs=logs)
        self.step += 1
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')

        self._print_schedule()

        for s,lr in self.lr_schedule:
            if self.step <= s:
                if self.prev_lr != lr:
                    self.prev_lr = lr
                    logger.info(f'\nStep {self.step}, updating learn rate: {lr}')
                tf.keras.backend.set_value(self.model.optimizer.lr, tf.keras.backend.get_value(lr))
                return

        logger.info(f"\n\n*** Maximum number of steps ({self.step}) exceeded. Stopping training\n")
        self.model.stop_training = True


    def on_train_batch_end(self, batch, logs=None):
        super().on_train_batch_end(batch=batch, logs=logs)
        logs = logs or {}
        logs["lr"] = tf.keras.backend.get_value(self.model.optimizer.lr)
        

    def _print_schedule(self):
        if not self.printed_schedule:
            self.printed_schedule = True 
            s = 'Learn rate schedule:\n'
            s += '  Less than step:\n'
            for step, lr in self.lr_schedule:
                s += f'  {step} --> {lr}\n'
            logger.info(s)

    

# https://keras.io/api/callbacks/early_stopping/
# If the validation accuracy doesn't improve after 'patience' epochs then stop training
# my_model.early_stopping = dict( 
#   monitor = 'val_accuracy',
#   patience = 35 
# )

# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard
# my_model.tensorboard = dict(
#     histogram_freq=0,       # frequency (in epochs) at which to compute activation and weight histograms 
#                             # for the layers of the model. If set to 0, histograms won't be computed. 
#                             # Validation data (or split) must be specified for histogram visualizations.
#     write_graph=False,       # whether to visualize the graph in TensorBoard. The log file can become quite large when write_graph is set to True.
#     write_images=False,     # whether to write model weights to visualize as image in TensorBoard.
#     update_freq="batch",    # 'batch' or 'epoch' or integer. When using 'batch', writes the losses and metrics 
#                             # to TensorBoard after each batch. The same applies for 'epoch'. 
#                             # If using an integer, let's say 1000, the callback will write the metrics and losses 
#                             # to TensorBoard every 1000 batches. Note that writing too frequently to 
#                             # TensorBoard can slow down your training.
#     profile_batch=(51,51),        # Profile the batch(es) to sample compute characteristics. 
#                             # profile_batch must be a non-negative integer or a tuple of integers. 
#                             # A pair of positive integers signify a range of batches to profile. 
#                             # By default, it will profile the second batch. Set profile_batch=0 to disable profiling.
# ) 

# NOTE: You can also add manually add other KerasCallbacks
# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/
# Any callbacks specified here will override the built-in callbacks 
# (e.g. my_model.reduce_lr_on_plateau, my_model.early_stopping)
my_model.train_callbacks = [
    tf.keras.callbacks.TerminateOnNaN(),
    LearnRateScheduleBatchCallback()
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
# Specify AudioFeatureGenerator Settings
# See https://siliconlabs.github.io/mltk/docs/audio/audio_feature_generator.html
#
frontend_settings = AudioFeatureGeneratorSettings()

frontend_settings.sample_rate_hz = 16000
frontend_settings.sample_length_ms = 1000
frontend_settings.window_size_ms = 30
frontend_settings.window_step_ms = 10
frontend_settings.filterbank_n_channels = 40 
frontend_settings.filterbank_upper_band_limit = 7500.0
frontend_settings.filterbank_lower_band_limit = 125.0
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

frontend_settings.activity_detection_enable = False # Enable the activity detection block
frontend_settings.activity_detection_alpha_a = 0.5
frontend_settings.activity_detection_alpha_b = 0.8
frontend_settings.activity_detection_arm_threshold = 0.75
frontend_settings.activity_detection_trip_threshold = 0.8

frontend_settings.dc_notch_filter_enable = True # Enable the DC notch filter
frontend_settings.dc_notch_filter_coefficient = 0.95

frontend_settings.quantize_dynamic_scale_enable = False # Enable dynamic quantization
frontend_settings.quantize_dynamic_scale_range_db = 40.0

# Add the Audio Feature generator settings to the model parameters
# This way, they are included in the generated .tflite model file
# See https://siliconlabs.github.io/mltk/docs/guides/model_parameters.html
my_model.model_parameters.update(frontend_settings)



##########################################################################
# Define the model architecture
#
def my_model_builder(model: MyModel):
    flags['batch_size'] = model.batch_size
    flags['sample_rate'] = frontend_settings.sample_rate_hz
    flags['window_size_ms'] = frontend_settings.window_size_ms
    flags['window_step_ms'] = frontend_settings.window_step_ms
    flags['clip_duration_ms'] = frontend_settings.sample_length_ms
    flags['mel_lower_edge_hertz'] = frontend_settings.filterbank_lower_band_limit
    flags['mel_upper_edge_hertz'] = frontend_settings.filterbank_upper_band_limit
    flags['mel_num_bins'] = frontend_settings.filterbank_n_channels
    flags['desired_samples'] = frontend_settings.sample_length

    keras_model = bc_resnet.model(flags)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True 
    )
    keras_model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(epsilon=1e-8),
        metrics=['accuracy']
    )

    return keras_model


my_model.build_model_function = my_model_builder
my_model.keras_custom_objects['TransitionBlock'] = bc_resnet.TransitionBlock
my_model.keras_custom_objects['NormalBlock'] = bc_resnet.NormalBlock
my_model.keras_custom_objects['SpeechFeatures'] = bc_resnet.speech_features.SpeechFeatures

# def my_model_saver(
#     mltk_model:MyModel, 
#     keras_model:tf.keras.Model, 
#     logger:logging.Logger
# ) -> tf.keras.Model:
#     return keras_model 

# my_model.on_save_keras_model = my_model_saver


##########################################################################
# Specify the other dataset settings
#

my_model.input_shape = frontend_settings.spectrogram_shape + (1,)

my_model.classes = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', '_silence_', '_unknown_']
my_model.class_weights = 'balanced' # Ensure the classes samples a balanced during training

unknown_class_percentage = 0.13
silence_class_percentage = 0.13
validation_split = 0.2

# Uncomment this to dump the augmented audio samples to the log directory
#data_dump_dir = my_model.create_log_dir('dataset_dump')


##########################################################################
# Create the audio augmentation pipeline
#
def audio_augmentation_pipeline_unaugmented(*args, **kwargs):
    return audio_augmentation_pipeline(*args, **kwargs, augment=False)

def audio_augmentation_pipeline(batch:np.ndarray, seed:np.ndarray, augment:bool=True) -> np.ndarray:
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
    y_shape = (batch_length, frontend_settings.sample_length,)
    y_batch = np.empty(y_shape, dtype=np.float32)
    
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
            offset=np.random.uniform(0, 1) if augment else 0
        )

        # Initialize the global audio augmentations instance
        # NOTE: We want this to be global so that we only initialize it once per subprocess
        if augment:
            audio_augmentations = globals().get('audio_augmentations', None)
            if audio_augmentations is None:
                dataset_dir = speech_commands_v2.load_clean_data()
                audio_augmentations = audiomentations.Compose(
                    p=0.90,
                    transforms=[ 
                    # audiomentations.PitchShift(min_semitones=-1, max_semitones=1, p=0.5),
                    audiomentations.TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
                    #audiomentations.Gain(min_gain_in_db=0.95, max_gain_in_db=1.5, p=0.75),
                    audiomentations.AddBackgroundNoise(
                        f'{dataset_dir}/_background_noise_', 
                        min_snr_in_db=20,
                        max_snr_in_db=40,
                        noise_rms="relative",
                        lru_cache_size=10,
                        p=0.75
                    ),
                    audiomentations.AddGaussianSNR(min_snr_in_db=30, max_snr_in_db=60, p=0.25),
                ])
                globals()['audio_augmentations'] = audio_augmentations

            # Apply random augmentations to the audio sample
            augmented_sample = audio_augmentations(adjusted_sample, original_sample_rate)

        else:
            augmented_sample = adjusted_sample

        # Convert the sample rate (if necessary)
        if original_sample_rate != frontend_settings.sample_rate_hz:
            augmented_sample = audio_utils.resample(
                augmented_sample, 
                orig_sr=original_sample_rate, 
                target_sr=frontend_settings.sample_rate_hz
            )

        # # Generate a spectrogram from the augmented audio sample
        # spectrogram = audio_utils.apply_frontend(
        #     sample=augmented_sample, 
        #     settings=frontend_settings, 
        #     dtype=np.float32
        # )
        # # The output spectrogram is 2D, add a channel dimension to make it 3D:
        # # (height, width, channels=1)
        # spectrogram = np.expand_dims(spectrogram, axis=-1)

        # # Dump the augmented audio sample AND corresponding spectrogram (if necessary)
        # data_dump_dir = globals().get('data_dump_dir', None)
        # if data_dump_dir:
        #     audio_dump_path = audio_utils.write_audio_file(
        #         data_dump_dir, 
        #         augmented_sample, 
        #         sample_rate=frontend_settings.sample_rate_hz
        #     )
        #     image_dump_path = audio_dump_path.replace('.wav', '.jpg')
        #     image_utils.write_image_file(image_dump_path, spectrogram)
            
        y_batch[i] = augmented_sample

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
        dataset_dir = speech_commands_v2.load_clean_data()

        # Create a tf.data.Dataset from the extracted "Speech Commands" directory
        max_samples_per_class = my_model.batch_size if test else -1
        features_ds, labels_ds = tf_dataset_utils.load_audio_directory(
            directory=dataset_dir,
            classes=my_model.classes,
            onehot_encode=False, # We're using categorical cross-entropy so one-hot encode the labels
            shuffle=True,
            seed=42,
            max_samples_per_class=max_samples_per_class,
            split=split,
            unknown_class_percentage=unknown_class_percentage,
            silence_class_percentage=silence_class_percentage,
            return_audio_data=False, # We only want to return the file paths
            class_counts=my_model.class_counts[subset], 
            list_valid_filenames_in_directory_function=speech_commands_v2.list_valid_filenames_in_directory
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
        labels_ds = labels_ds.batch(per_job_batch_size // 100, drop_remainder=True)
        features_ds, pool = tf_dataset_utils.parallel_process(
            features_ds,
            audio_augmentation_pipeline if subset == 'training' else audio_augmentation_pipeline_unaugmented,
            dtype=np.float32,
            n_jobs=72 if subset == 'training' else 32, # These are the settings for a 128 CPU core cloud machine
            #n_jobs=64 if subset == 'training' else 22, # These are the settings for a 96 CPU core cloud machine
            #n_jobs=.65 if subset == 'training' else .35,
            name=subset,
        )
        self.pools.append(pool)
        features_ds = features_ds.unbatch()
        labels_ds = labels_ds.unbatch()

        # Pre-fetching batches can help with throughput
        features_ds = features_ds.prefetch(per_job_batch_size)

        # Combine the augmented audio samples with their corresponding labels
        ds = tf.data.Dataset.zip((features_ds, labels_ds))

        # Shuffle the data for each sample
        # A perfect shuffle would use n_samples but this can slow down training,
        # so we just shuffle batches of the data
        #ds = ds.shuffle(per_job_batch_size, reshuffle_each_iteration=True)
        
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
