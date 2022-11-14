import pytest

from mltk.utils.test_helper import run_model_operation, generate_run_model_params

@pytest.mark.parametrize(*generate_run_model_params())
def test_keyword_spotting_on_off(op, tflite, build):
    run_model_operation('keyword_spotting_on_off', op, tflite, build)

@pytest.mark.parametrize(*generate_run_model_params())
def test_keyword_spotting_mobilenetv2(op, tflite, build):
    run_model_operation('keyword_spotting_mobilenetv2', op, tflite, build)

@pytest.mark.parametrize(*generate_run_model_params())
def test_keyword_spotting_with_transfer_learning(op, tflite, build):
    run_model_operation('keyword_spotting_with_transfer_learning', op, tflite, build)

@pytest.mark.parametrize(*generate_run_model_params())
def test_rock_paper_scissors(op, tflite, build):
    run_model_operation('rock_paper_scissors', op, tflite, build)

@pytest.mark.parametrize(*generate_run_model_params(train=False, evaluate=False, quantize=False, build=False, tflite=True, run_directly=False))
def test_fingerprint_signature_generator(op, tflite, build):
    run_model_operation('fingerprint_signature_generator', op, tflite, build)

@pytest.mark.parametrize(*generate_run_model_params())
def test_keyword_spotting_pacman(op, tflite, build):
    run_model_operation('keyword_spotting_pacman', op, tflite, build)


@pytest.mark.parametrize(*generate_run_model_params())
def test_keyword_spotting_pacman_v2(op, tflite, build):
    run_model_operation('keyword_spotting_pacman_v2', op, tflite, build)