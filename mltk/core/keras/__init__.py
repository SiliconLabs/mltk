

from keras_preprocessing import get_keras_submodule

try:
    DataSequence = get_keras_submodule('utils').Sequence
except:
    try:
        # This is a work-around for TF-2.9
        from keras.utils.data_utils import Sequence
        DataSequence = Sequence
    except:
        DataSequence = object