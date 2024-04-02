import sys
import os
import pytest
from mltk.utils.test_helper import run_mltk_command




def test_build_gsdk_mltk_extension():
    run_mltk_command('build', 'gsdk_mltk_extension', '--no-show')


def test_build_docs():
    curdir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
    requirements_path = f'{curdir}/../utils/dev_requirements.txt'
    run_mltk_command('-m', 'pip', 'install', '-r', requirements_path, exe_path=sys.executable)
    run_mltk_command('build', 'docs', '--no-show', '--no-checklinks')
    run_mltk_command('build', 'docs', '--revert-only')


# Skip for now since the wrapper is loaded into the python environment and cannot be built as a unit test
@pytest.mark.skip() 
def test_build_audio_feature_generator_wrapper():
    run_mltk_command('build', 'audio_feature_generator_wrapper')

# Skip for now since the wrapper is loaded into the python environment and cannot be built as a unit test
@pytest.mark.skip() 
def test_build_mvp_wrapper():
    run_mltk_command('build', 'mvp_wrapper')

# Skip for now since the wrapper is loaded into the python environment and cannot be built as a unit test
@pytest.mark.skip() 
def test_build_tflite_micro_wrapper():
    run_mltk_command('build', 'tflite_micro_wrapper')


@pytest.mark.skip()
def test_build_python_package():
    run_mltk_command('build', 'python_package')