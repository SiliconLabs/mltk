import os
import sys

mltk_models_shared_dir = os.path.normpath(os.path.dirname(os.path.abspath(__file__)) + '/..')
if mltk_models_shared_dir not in sys.path:
    sys.path.append(mltk_models_shared_dir)