import pytest
from mltk.utils.test_helper import run_mltk_command
from mltk.utils.commander import query_platform


def test_commander_help():
    run_mltk_command('commander', '--help')

def test_commander_device_info():
    try:
        query_platform()
    except:
        pytest.skip('No device connected')
        return
    
    try:
        run_mltk_command('commander', 'device', 'info', '-d', 'efr32')
    except RuntimeError:
        run_mltk_command('commander', 'device', 'info', '-d', 'efm32')

