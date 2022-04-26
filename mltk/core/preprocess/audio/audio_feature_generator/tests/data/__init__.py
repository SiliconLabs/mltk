


from mltk.core.preprocess.audio.audio_feature_generator import AudioFeatureGeneratorSettings

from .yes_data import (YES_INPUT_AUDIO, YES_OUTPUT_FEATURES_INT8)
from .no_data import (NO_INPUT_AUDIO, NO_OUTPUT_FEATURES_INT8)

# The values come from:
# https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/micro_speech/micro_features/micro_features_generator.cc
DEFAULT_SETTINGS = AudioFeatureGeneratorSettings()
DEFAULT_SETTINGS.sample_rate_hz = 16000
DEFAULT_SETTINGS.sample_length_ms = 1000
DEFAULT_SETTINGS.window_size_ms = 30
DEFAULT_SETTINGS.window_step_ms = 20

DEFAULT_SETTINGS.filterbank_n_channels = 40

DEFAULT_SETTINGS.filterbank_lower_band_limit = 125
DEFAULT_SETTINGS.filterbank_upper_band_limit = 7500

DEFAULT_SETTINGS.noise_reduction_enable = True 
DEFAULT_SETTINGS.noise_reduction_smoothing_bits = 10 
DEFAULT_SETTINGS.noise_reduction_even_smoothing = 0.025
DEFAULT_SETTINGS.noise_reduction_odd_smoothing = 0.06
DEFAULT_SETTINGS.noise_reduction_min_signal_remaining = 0.05

DEFAULT_SETTINGS.pcan_enable = True
DEFAULT_SETTINGS.pcan_strength = 0.95
DEFAULT_SETTINGS.pcan_offset = 80
DEFAULT_SETTINGS.pcan_gain_bits = 21

DEFAULT_SETTINGS.log_scale_enable = True
DEFAULT_SETTINGS.log_scale_shift = 6