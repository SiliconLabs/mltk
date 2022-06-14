
import pytest
from mltk.utils.test_helper import run_model_operation, generate_run_model_params

@pytest.mark.parametrize(*generate_run_model_params())
def test_binary_classification(op, tflite, build):
    run_model_operation('binary_classification-test', op, tflite, build)

@pytest.mark.parametrize(*generate_run_model_params())
def test_image_example1(op, tflite, build):
    run_model_operation('image_example1-test', op, tflite, build)

@pytest.mark.parametrize(*generate_run_model_params())
def test_audio_example1(op, tflite, build):
    run_model_operation('audio_example1-test', op, tflite, build)

@pytest.mark.parametrize(*generate_run_model_params())
def test_conv1d_example(op, tflite, build):
    run_model_operation('conv1d_example-test', op, tflite, build)

@pytest.mark.parametrize(*generate_run_model_params())
def test_siamese_contrastive_example(op, tflite, build):
    run_model_operation('siamese_contrastive-test', op, tflite, build)

@pytest.mark.parametrize(*generate_run_model_params())
def test_autoencoder_example(op, tflite, build):
    run_model_operation('autoencoder_example-test', op, tflite, build)

