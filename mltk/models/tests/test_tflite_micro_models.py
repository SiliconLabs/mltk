
import pytest
from mltk.utils.test_helper import run_model_operation, generate_run_model_params


@pytest.mark.parametrize(*generate_run_model_params())
def test_tflite_micro_speech(op, tflite, build):
    run_model_operation('tflite_micro_speech', op, tflite, build)

@pytest.mark.parametrize(*generate_run_model_params())
def test_tflite_micro_magic_wand(op, tflite, build):
    run_model_operation('tflite_micro_magic_wand', op, tflite, build)