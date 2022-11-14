import pytest 

from mltk import MLTK_DIR
from mltk.utils.test_helper import run_mltk_command
from mltk.utils.test_helper.data import IMAGE_EXAMPLE1_TFLITE_PATH
from mltk.utils.path import create_tempdir, remove_directory
from mltk.utils.commander import query_platform


def test_profile_help():
    run_mltk_command('profile', '--help')


def test_profile_model_name():
    run_mltk_command('profile', 'image_example1')

def test_profile_model_name_mvp():
    run_mltk_command('profile', 'image_example1', '--accelerator', 'mvp')

def test_profile_archive_path():
    path = f'{MLTK_DIR}/models/examples/image_example1.mltk.zip'
    run_mltk_command('profile', path)

def test_profile_tflite_path():
    run_mltk_command('profile', IMAGE_EXAMPLE1_TFLITE_PATH)

def test_profile_build():
    run_mltk_command('profile', 'image_example1', '--build')

def test_profile_output():
    out_dir = create_tempdir('tests/profile_output')
    remove_directory(out_dir)
    run_mltk_command('profile', 'image_example1', '--output', out_dir)
    remove_directory(out_dir)

def test_profile_noformat():
    run_mltk_command('profile', 'image_example1', '--no-format')

def test_profile_device():
    try:
        query_platform()
    except:
        pytest.skip('No device connected')
        return

    run_mltk_command('profile', 'image_example1', '--device')