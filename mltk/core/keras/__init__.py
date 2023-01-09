
# Tensorflow keeps moving these packages ...
# To try to import based on the current TF version


try:
    from keras_preprocessing import get_keras_submodule
    DataSequence = get_keras_submodule('utils').Sequence
except ModuleNotFoundError:
    try:
        from tensorflow.keras.utils.data_utils import Sequence
        DataSequence = Sequence
    except ModuleNotFoundError:
        try:
            from keras.utils.data_utils import Sequence
            DataSequence = Sequence
        except:
            DataSequence = object

try:
    from keras_preprocessing.image import ImageDataGenerator
except ModuleNotFoundError:
    try:
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
    except ModuleNotFoundError:
        from keras.preprocessing.image import ImageDataGenerator

try:
    from keras_preprocessing.image.utils import (array_to_img, img_to_array, load_img)
except ModuleNotFoundError:
    try:
        from tensorflow.keras.preprocessing.image import (array_to_img, img_to_array, load_img)
    except ModuleNotFoundError:
        from keras.preprocessing.image import (array_to_img, img_to_array, load_img)


try:
    from keras_preprocessing.image.iterator import Iterator as ImageIterator
except ModuleNotFoundError:
    try:
        from tensorflow.keras.preprocessing.image import Iterator as ImageIterator
    except ModuleNotFoundError:
        from keras.preprocessing.image import Iterator as ImageIterator



