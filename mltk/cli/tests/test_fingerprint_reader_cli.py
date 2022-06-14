import pytest
from mltk import MLTK_DIR
from mltk.utils.test_helper import run_mltk_command
from mltk.utils.commander import query_platform



def test_help():
    run_mltk_command('fingerprint_reader', '--help')


def test_with_model_name():
    try:
        query_platform()
    except:
        pytest.skip('No embedded device connected')
        return 
    run_mltk_command('fingerprint_reader', 'fingerprint_signature_generator', '--test')
