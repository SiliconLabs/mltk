
import numpy as np
from mltk.core.preprocess.audio.audio_feature_generator import AudioFeatureGenerator
from mltk.core.preprocess.audio.audio_feature_generator.tests.data import (
    DEFAULT_SETTINGS, 
    YES_INPUT_AUDIO, 
    YES_OUTPUT_FEATURES_INT8,
    NO_INPUT_AUDIO,
    NO_OUTPUT_FEATURES_INT8
)


def test_yes_samples():
    settings = DEFAULT_SETTINGS
    mfe = AudioFeatureGenerator(settings)
    sample = np.asarray(YES_INPUT_AUDIO, dtype=np.int16)
    calculated = mfe.process_sample(sample, dtype=np.int8)
    
    expected = np.reshape(np.array(YES_OUTPUT_FEATURES_INT8, dtype=np.int8), settings.spectrogram_shape)

    assert np.allclose(calculated, expected)


def test_no_samples():
    settings = DEFAULT_SETTINGS
    mfe = AudioFeatureGenerator(settings)
    sample = np.asarray(NO_INPUT_AUDIO, dtype=np.int16)
    calculated = mfe.process_sample(sample, dtype=np.int8)
    
    expected = np.reshape(np.array(NO_OUTPUT_FEATURES_INT8, dtype=np.int8), settings.spectrogram_shape)

    assert np.allclose(calculated, expected)

