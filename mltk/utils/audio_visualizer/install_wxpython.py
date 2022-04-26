import logging 
from mltk.utils.python import install_pip_package


def install_wxpython(logger:logging.Logger=None):
    install_pip_package('wxpython>=4.1.1,<4.2', module_name='wx', logger=logger)
