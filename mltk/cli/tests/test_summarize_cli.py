
from mltk import MLTK_DIR
from mltk.utils.test_helper import run_mltk_command
from mltk.utils.test_helper.data import IMAGE_EXAMPLE1_TFLITE_PATH, IMAGE_EXAMPLE1_H5_PATH
from mltk.utils.path import create_tempdir


def test_summarize_help():
    run_mltk_command('summarize', '--help')


def test_summarize_model_name():
    run_mltk_command('summarize', 'image_example1')

def test_summarize_model_name_tflite():
    run_mltk_command('summarize', 'image_example1')

def test_summarize_model_name_build():
    run_mltk_command('summarize', 'image_example1', '--build')

def test_summarize_model_name_build_tflite():
    run_mltk_command('summarize', 'image_example1', '--build', '--tflite')

def test_summarize_archive():
    archive_path = f'{MLTK_DIR}/models/examples/image_example1.mltk.zip'
    run_mltk_command('summarize', archive_path)

def test_summarize_archive_tflite():
    archive_path = f'{MLTK_DIR}/models/examples/image_example1.mltk.zip'
    run_mltk_command('summarize', archive_path, '--tflite')

def test_summarize_script():
    spec_path = f'{MLTK_DIR}/utils/test_helper/test_image_model.py'
    run_mltk_command('summarize', spec_path, '--build')

def test_summarize_script_tflite():
    spec_path = f'{MLTK_DIR}/utils/test_helper/test_image_model.py'
    run_mltk_command('summarize', spec_path, '--build', '--tflite')

def test_summarize_tflite():
    run_mltk_command('summarize', IMAGE_EXAMPLE1_TFLITE_PATH)

def test_summarize_h5():
    run_mltk_command('summarize', IMAGE_EXAMPLE1_H5_PATH)

def test_summarize_output():
    out_path = create_tempdir('tests') + '/summary_test.txt'
    run_mltk_command('summarize', 'image_example1', '--output', out_path)