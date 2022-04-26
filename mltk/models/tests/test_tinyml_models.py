import pytest
from mltk.utils.test_helper import run_model_operation, generate_run_model_params


@pytest.mark.parametrize(*generate_run_model_params())
def test_anomaly_detection(op, tflite, build):
    run_model_operation('anomaly_detection-test', op, tflite, build)

@pytest.mark.parametrize(*generate_run_model_params())
def test_image_classification(op, tflite, build):
    run_model_operation('image_classification-test', op, tflite, build)

@pytest.mark.parametrize(*generate_run_model_params())
def test_keyword_spotting(op, tflite, build):
    run_model_operation('keyword_spotting-test', op, tflite, build)

@pytest.mark.parametrize(*generate_run_model_params())
def test_visual_wake_words(op, tflite, build):
    run_model_operation('visual_wake_words-test', op, tflite, build)