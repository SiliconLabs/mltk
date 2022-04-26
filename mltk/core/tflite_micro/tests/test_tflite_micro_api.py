
import os

import numpy as np
from mltk.core import TfliteModel
from mltk.core.tflite_micro import TfliteMicro, TfliteMicroModel
from mltk.core.tflite_micro.tflite_micro_accelerator import TfliteMicroAccelerator
from mltk.utils.test_helper.data import (TFLITE_MICRO_SPEECH_TFLITE_PATH, IMAGE_EXAMPLE1_TFLITE_PATH)



def test_git_hash():
    v = TfliteMicro.git_hash()
    assert isinstance(v, str)
    assert len(v) > 1

def test_api_version():
    v = TfliteMicro.api_version()
    assert v == 1

def test_set_log_level():
    orig_level = TfliteMicro.get_log_level()
    levels = ['debug', 'info', 'warn', 'error']

    for l in levels:
        TfliteMicro.set_log_level(l)
        assert TfliteMicro.get_log_level() == l
    TfliteMicro.set_log_level(orig_level)

def test_normalize_accelerator_name():
    assert TfliteMicro.normalize_accelerator_name(None) is None
    assert TfliteMicro.normalize_accelerator_name('bogus') is None
    assert TfliteMicro.normalize_accelerator_name('mvp') == 'MVP'

def test_get_supported_accelerators():
    accs  = TfliteMicro.get_supported_accelerators()
    assert set(accs) == {'MVP'}

def test_accelerator_is_supported():
    assert TfliteMicro.accelerator_is_supported(None) is False
    assert TfliteMicro.accelerator_is_supported('bogus') is False
    assert TfliteMicro.accelerator_is_supported('mvp') is True

def test_load_tflite_model():
    tflm_model = TfliteMicro.load_tflite_model(IMAGE_EXAMPLE1_TFLITE_PATH)
    assert isinstance(tflm_model, TfliteMicroModel)
    TfliteMicro.unload_model(tflm_model)

    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_EXAMPLE1_TFLITE_PATH)
    tflm_model = TfliteMicro.load_tflite_model(tflite_model)
    assert isinstance(tflm_model, TfliteMicroModel)
    TfliteMicro.unload_model(tflm_model)


def test_load_tflite_model_mvp():
    tflm_model = TfliteMicro.load_tflite_model(
        IMAGE_EXAMPLE1_TFLITE_PATH, 
        accelerator='mvp'
    )
    assert isinstance(tflm_model, TfliteMicroModel)
    assert isinstance(tflm_model.accelerator, TfliteMicroAccelerator)
    assert tflm_model.accelerator.name == 'MVP'
    TfliteMicro.unload_model(tflm_model)


def test_profile_model():
    results = TfliteMicro.profile_model(IMAGE_EXAMPLE1_TFLITE_PATH)
    assert results.n_layers == 8
    assert results.accelerator_cycles == 0
    assert results.macs > 0


def test_profile_model_mvp():
    results = TfliteMicro.profile_model(IMAGE_EXAMPLE1_TFLITE_PATH, accelerator='mvp')
    assert results.n_layers == 8
    assert results.accelerator_cycles > 0
    assert results.cpu_cycles > 0
    assert results.energy > 0
    assert results.macs > 0

def test_profile_model_mvp_unsupported_layer():
    results = TfliteMicro.profile_model(TFLITE_MICRO_SPEECH_TFLITE_PATH, accelerator='mvp')
    assert results.n_layers == 4
    assert results.n_unsupported_layers == 1
    assert results.accelerator_cycles > 0
    assert results.cpu_cycles > 0
    assert results.energy > 0
    assert results.macs > 0


def test_record_model():
    input_data = np.random.uniform(low=-127, high=128, size=(96,96,1)).astype(np.int8)
    layers = TfliteMicro.record_model(IMAGE_EXAMPLE1_TFLITE_PATH, input_data)
    assert len(layers) == 8