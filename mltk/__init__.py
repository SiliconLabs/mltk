
import sys
import os

__version__ = '0.14.0'

MLTK_DIR = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
MLTK_ROOT_DIR = os.path.dirname(MLTK_DIR).replace('\\', '/')


def disable_tensorflow():
    """Disable the Tensorflow Python package with a placeholder

    Tensorflow is very bloaty
    If we can get away without importing it we can save a lot of time
    and potentially diskspace if we remove it as a dependency

    This also disables matplotlib which can also be bloaty
    """
    sys.path.insert(0, f'{MLTK_DIR}/core/keras/tensorflow_placeholder')


if os.environ.get('MLTK_DISABLE_TF', '0') == '1':
    disable_tensorflow()
