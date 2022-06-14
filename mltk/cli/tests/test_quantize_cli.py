import os
import pytest

from mltk import MLTK_DIR
from mltk.utils.test_helper import run_mltk_command
from mltk.utils.path import create_tempdir


archive_path = f'{MLTK_DIR}/utils/test_helper/test_image_model.mltk.zip'


@pytest.mark.dependency()
def test_train_model():
    if os.path.exists(archive_path):
        os.remove(archive_path)
    run_mltk_command('train', 'test_image_model', '--clean', '-v', '--no-evaluate', update_model_path=True)



def test_quantize_help():
    run_mltk_command('quantize', '--help')


@pytest.mark.dependency(depends=['test_train_model'])
def test_quantize_model_name():
    run_mltk_command('quantize', 'test_image_model', update_model_path=True)

@pytest.mark.dependency(depends=['test_train_model'])
def test_quantize_model_name_output_path():
    tflite_path = create_tempdir('tests/tmp') + '/test.tflite'
    if os.path.exists(tflite_path):
        os.remove(tflite_path)
    run_mltk_command('quantize', 'test_image_model', '--output', tflite_path, update_model_path=True)
    assert os.path.exists(tflite_path)

@pytest.mark.dependency(depends=['test_train_model'])
def test_quantize_model_name_output_dir():
    tflite_dir = create_tempdir('tests/tmp')
    tflite_path = tflite_dir + '/test_image_model.tflite'
    if os.path.exists(tflite_path):
        os.remove(tflite_path)
    run_mltk_command('quantize', 'test_image_model', '--output', tflite_dir, update_model_path=True)
    assert os.path.exists(tflite_path)

@pytest.mark.dependency(depends=['test_train_model'])
def test_quantize_model_name_build():
    run_mltk_command('quantize', 'test_image_model', '--build', update_model_path=True)

@pytest.mark.dependency(depends=['test_train_model'])
def test_quantize_archive():
    run_mltk_command('quantize', archive_path)

@pytest.mark.dependency(depends=['test_train_model'])
def test_quantize_archive_output():
    run_mltk_command('quantize', archive_path, '--output', os.path.dirname(archive_path))

@pytest.mark.dependency(depends=['test_train_model'])
def test_quantize_spec():
    spec_path = f'{MLTK_DIR}/utils/test_helper/test_image_model.py'
    run_mltk_command('quantize', spec_path, '--build', '--output', os.path.dirname(spec_path))