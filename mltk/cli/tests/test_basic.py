
import mltk
from mltk.utils.test_helper import run_mltk_command


def test_version():
    retmsg = run_mltk_command('--version')
    assert retmsg.strip() == mltk.__version__


def test_help():
    run_mltk_command( '--help')


        