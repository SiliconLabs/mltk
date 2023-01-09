"""Fashion-MNIST
****************************************

This is a dataset of 60,000 28x28 grayscale images of 10 fashion categories,
along with a test set of 10,000 images. This dataset can be used as
a drop-in replacement for MNIST.

The classes are:  

- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

Returns:
Tuple of NumPy arrays: ``(x_train, y_train), (x_test, y_test)``.

**x_train**: uint8 NumPy array of grayscale image data with shapes
``(60000, 28, 28)``, containing the training data.

**y_train**: uint8 NumPy array of labels (integers in range 0-9)
with shape ``(60000,)`` for the training data.

**x_test**: uint8 NumPy array of grayscale image data with shapes
(10000, 28, 28), containing the test data.

**y_test**: uint8 NumPy array of labels (integers in range 0-9)
with shape ``(10000,)`` for the test data.

Example:

.. code-block::

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)


License:
  The copyright for Fashion-MNIST is held by Zalando SE.
  Fashion-MNIST is licensed under the `MIT license <https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE>`_

"""

import os
import gzip
from typing import Tuple
import numpy as np
from mltk.utils.path import create_user_dir
from mltk.utils.archive_downloader import download_verify_extract
from mltk.utils.logger import get_logger
from mltk.core.keras import array_to_img


INPUT_SHAPE = (28,28)
CLASSES = [
    'tshirt', 
    'trouser', 
    'pullover', 
    'dress', 
    'coat', 
    'sandal', 
    'shirt', 
    'sneaker', 
    'bag', 
    'boot'
]




def load_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Download the dataset, extract, load into memory, 
    and return as a tuple of numpy arrays
    
    Returns:
        Tuple of NumPy arrays: ``(x_train, y_train), (x_test, y_test)``
    """
    y_train_path = download_verify_extract(
        url='https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz',
        file_hash='09814CFEF5A041118CEACE42F8DAE995319D331A',
        show_progress=True,
        extract=False
    )
    x_train_path = download_verify_extract(
        url='https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz',
        file_hash='95978B76B6897F6CA69A25145D01716EFB615989',
        show_progress=True,
        extract=False
    )
    y_test_path = download_verify_extract(
        url='https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz',
        file_hash='9CAAD14E1AFF9ADAC77D3744963212D36AF15BEE',
        show_progress=True,
        extract=False
    )
    x_test_path = download_verify_extract(
        url='https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz',
        file_hash='5EDDA96C6D8C36FF915115A0E8136D370A021576',
        show_progress=True,
        extract=False
    )

    with gzip.open(y_train_path, 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(x_train_path, 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(y_test_path, 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(x_test_path, 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)


def load_data_directory() -> str:
    """Download the dataset, extract all sample images to a directory, 
    and return the path to the directory.

    Each sample type is extract to its corresponding subdirectory, e.g.:

    ~/.mltk/datasets/fashion_mnist/tshirt
    ~/.mltk/datasets/fashion_mnist/dress
    ...
    
    Returns:
        Path to extract directory:
    """

    dataset_dir = f'{create_user_dir()}/datasets/fashion_mnist'


    (x_train, y_train), (x_test, y_test) = load_data()
    x_samples = np.concatenate((x_train, x_test))
    y_samples = np.concatenate((y_train, y_test))

    class_ids, class_counts = np.unique(y_samples, return_counts=True)

    expected_class_counts = {}
    for class_id, class_count in zip(class_ids, class_counts):
        expected_class_counts[CLASSES[class_id]] = class_count

    for class_id, class_label in enumerate(CLASSES):
        dataset_class_dir = f'{dataset_dir}/{class_label}'
        os.makedirs(dataset_class_dir, exist_ok=True)
        class_count = len(os.listdir(dataset_class_dir))

        if class_count != expected_class_counts[class_label]:
            get_logger().warning(f'Generating {dataset_class_dir}')
            sample_count = 0
            for x, y in zip(x_samples, y_samples):
                if class_id != y:
                    continue
                sample_path = f'{dataset_class_dir}/{sample_count}.jpg'
                sample_count += 1

                x = np.expand_dims(x, axis=-1)
                img = array_to_img(x, scale=False, dtype='uint8')
                img.save(sample_path)


    return dataset_dir