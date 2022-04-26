
from keras_preprocessing import get_keras_submodule

try:
    DataSequence = get_keras_submodule('utils').Sequence
except ImportError:
    DataSequence = object