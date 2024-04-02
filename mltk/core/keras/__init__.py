
# Tensorflow keeps moving these packages ...
# To try to import based on the current TF version

DataSequence = object
KerasModel = object 
ImageIterator = object
load_keras_model = None 
array_to_img = None 
img_to_array = None 
load_img = None 


try:
    from tf_keras.utils import Sequence
    from tf_keras.preprocessing.image import ImageDataGenerator
    from tf_keras.preprocessing.image import (array_to_img, img_to_array, load_img)
    from tf_keras.preprocessing.image import Iterator as ImageIterator
    from tf_keras.models import Model as KerasModel
    from tf_keras.models import load_model as load_keras_model
    DataSequence = Sequence
except (ModuleNotFoundError, ImportError):
    try:
        from tensorflow.keras.utils.data_utils import Sequence
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.preprocessing.image import (array_to_img, img_to_array, load_img)
        from tensorflow.keras.preprocessing.image import Iterator as ImageIterator
        from tensorflow.keras.models import load_model as load_keras_model
        from tensorflow.keras.models import Model as KerasModel
        DataSequence = Sequence
    except (ModuleNotFoundError, ImportError):
        try:
            from keras.utils import Sequence
            from keras.preprocessing.image import ImageDataGenerator
            from keras.preprocessing.image import (array_to_img, img_to_array, load_img)
            from keras.preprocessing.image import Iterator as ImageIterator
            from keras.models import Model as KerasModel
            from keras.models import load_model as load_keras_model
            DataSequence = Sequence
        except (ModuleNotFoundError, ImportError):
            ...