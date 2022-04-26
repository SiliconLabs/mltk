import pytest
from mltk import MLTK_DIR
from mltk.utils.test_helper import run_mltk_command
from mltk.utils.commander import query_platform



def test_help():
    run_mltk_command('classify_audio', '--help')


def test_pc_name():
    run_mltk_command('classify_audio', 'tflite_micro_speech', '--test')


def test_pc_archive():
    archive_path = f'{MLTK_DIR}/models/tflite_micro/tflite_micro_speech.mltk.zip'
    run_mltk_command('classify_audio', archive_path, '--test')


def test_device():
    try:
        query_platform()
    except:
        pytest.skip('No embedded device connected')
        return 

    run_mltk_command('classify_audio', 'tflite_micro_speech', '--device', '--test')
