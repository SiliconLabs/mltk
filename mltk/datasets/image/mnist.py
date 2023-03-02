"""MNIST
****************************************

This is a dataset of 60,000 28x28 grayscale images of the 10 digits,
along with a test set of 10,000 images.
More info can be found at the
`MNIST homepage <http://yann.lecun.com/exdb/mnist/>`_



Example
----------

.. code-block::

  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  assert x_train.shape == (60000, 28, 28)
  assert x_test.shape == (10000, 28, 28)
  assert y_train.shape == (60000,)
  assert y_test.shape == (10000,)



License
-----------

  Yann LeCun and Corinna Cortes hold the copyright of MNIST dataset,
  which is a derivative work from original NIST datasets.
  MNIST dataset is made available under the terms of the
  `Creative Commons Attribution-Share Alike 3.0 license. <https://creativecommons.org/licenses/by-sa/3.0/>`_

"""
import os
import logging
from typing import Tuple
import numpy as np
from mltk.utils.path import create_user_dir
from mltk.utils.archive_downloader import download_verify_extract
from mltk.utils.logger import get_logger
from mltk.core.keras import array_to_img


INPUT_SHAPE = (28, 28)
"""The shape of each sample"""
CLASSES = [
  '0',
  '1',
  '2',
  '3',
  '4',
  '5',
  '6',
  '7',
  '8',
  '9'
]
"""Labels for dataset samples"""


DOWNLOAD_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
"""Public download URL"""
VERIFY_SHA1 = '731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1'
"""SHA1 hash of archive file"""


def load_data(
    dest_dir:str=None,
    dest_subdir='datasets/mnist',
    logger:logging.Logger=None,
    clean_dest_dir=False
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Download the dataset, extract, load into memory,
    and return as a tuple of numpy arrays


    Returns:
        Tuple of NumPy arrays: ``(x_train, y_train), (x_test, y_test)``

      **x_train**: uint8 NumPy array of grayscale image data with shapes
        ``(60000, 28, 28)``, containing the training data. Pixel values range
        from 0 to 255.

      **y_train**: uint8 NumPy array of digit labels (integers in range 0-9)
        with shape ``(60000,)`` for the training data.

      **x_test**: uint8 NumPy array of grayscale image data with shapes
        (10000, 28, 28), containing the test data. Pixel values range
        from 0 to 255.

      **y_test**: uint8 NumPy array of digit labels (integers in range 0-9)
        with shape ``(10000,)`` for the test data.
    """

    if dest_dir:
        dest_subdir = None

    path = download_verify_extract(
        url=DOWNLOAD_URL,
        file_hash=VERIFY_SHA1,
        dest_dir=dest_dir,
        dest_subdir=dest_subdir,
        show_progress=True,
        extract=False,
        logger=logger,
        clean_dest_dir=clean_dest_dir
    )
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    return (x_train, y_train), (x_test, y_test)


def load_data_directory(
    dest_dir:str=None,
    dest_subdir='datasets/mnist',
    logger:logging.Logger=None,
    clean_dest_dir=False
) -> str:
    """Download the dataset, extract all sample images to a directory,
    and return the path to the directory.

    Each sample type is extract to its corresponding subdirectory, e.g.:

    ~/.mltk/datasets/mnist/0
    ~/.mltk/datasets/mnist/1
    ...

    Returns:
        Path to extract directory:
    """

    if not dest_dir:
        dataset_dir = f'{create_user_dir()}/{dest_subdir}'

    (x_train, y_train), (x_test, y_test) = load_data(
        dest_dir=dest_dir,
        logger=logger,
        clean_dest_dir=clean_dest_dir
    )
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