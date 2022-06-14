import pytest
from mltk.utils.test_helper import run_mltk_command


def test_build_audio_feature_generator_wrapper():
    run_mltk_command('build', 'audio_feature_generator_wrapper')


def test_build_docs():
    run_mltk_command('build', 'docs', '--no-show', '--no-checklinks')
    run_mltk_command('build', 'docs', '--revert-only')


def test_build_gsdk_mltk_extension():
    run_mltk_command('build', 'gsdk_mltk_extension', '--no-show')


def test_build_mvp_wrapper():
    run_mltk_command('build', 'mvp_wrapper')


def test_build_python_package():
    run_mltk_command('build', 'python_package')


def test_build_tflite_micro_wrapper():
    run_mltk_command('build', 'tflite_micro_wrapper')

