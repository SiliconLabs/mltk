import sys
import os

def path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)) + '/..')
    return os.path.join(base_path, relative_path)