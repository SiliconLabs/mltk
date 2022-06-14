import os
import pytest
from mltk import MLTK_DIR
from mltk.utils.test_helper import run_mltk_command
from mltk.utils.test_helper.data import (
    IMAGE_CLASSIFICATION_TFLITE_PATH, 
    IMAGE_EXAMPLE1_H5_PATH
)


archive_path = f'{MLTK_DIR}/utils/test_helper/test_image_model.mltk.zip'


@pytest.mark.dependency()
def test_train_model():
    if os.path.exists(archive_path):
        os.remove(archive_path)
    run_mltk_command('train', 'test_image_model', '--clean', '-v', '--no-evaluate', update_model_path=True)


def test_help():
    run_mltk_command('view', '--help')

def test_name():
    run_mltk_command('view', 'image_example1')

def test_name_tflite():
    run_mltk_command('view', 'image_example1', '--tflite')

def test_name_build():
    run_mltk_command('view', 'image_example1', '--build')

def test_name_build_tflite():
    run_mltk_command('view', 'image_example1', '--build', '--tflite')

def test_tflite():
    run_mltk_command('view', IMAGE_CLASSIFICATION_TFLITE_PATH)

def test_h5():
    run_mltk_command('view', IMAGE_EXAMPLE1_H5_PATH)

def test_script():
    spec_path = f'{MLTK_DIR}/utils/test_helper/test_image_model.py'
    run_mltk_command('view', spec_path, '--build')

def test_script_tflite():
    spec_path = f'{MLTK_DIR}/utils/test_helper/test_image_model.py'
    run_mltk_command('view', spec_path, '--build', '--tflite')

@pytest.mark.dependency(depends=['test_train_model'])
def test_archive():
    
    run_mltk_command('view', archive_path)

@pytest.mark.dependency(depends=['test_train_model'])
def test_archive_tflite():
    run_mltk_command('view', archive_path, '--tflite')


