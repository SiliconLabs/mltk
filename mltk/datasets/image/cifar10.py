"""CIFAR10
****************************************

This is a dataset of 50,000 32x32 color training images and 10,000 test
images, labeled over 10 categories. See more info at the
`CIFAR homepage <https://www.cs.toronto.edu/~kriz/cifar.html>`_

The classes are:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

Example
------------

.. code-block::

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert y_test.shape == (10000, 1)


"""
import os
from typing import Tuple
import numpy as np
import pickle
import logging

from tensorflow.keras import backend
from mltk.utils.archive_downloader import download_verify_extract
from mltk.utils.path import create_user_dir
from mltk.utils.logger import get_logger
from mltk.core.keras import array_to_img



INPUT_SHAPE = (32,32,3)
CLASSES = [
  'airplane',
  'automobile',
  'bird',
  'cat',
  'deer',
  'dog',
  'frog',
  'horse',
  'ship',
  'truck'
]
DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
"""Public download URL"""
VERIFY_SHA1 = '6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce'
"""SHA1 hash of archive file"""


def load_data(
    dest_dir:str=None,
    dest_subdir='datasets/cifar10',
    logger:logging.Logger=None,
    clean_dest_dir=False
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Download the dataset, extract, load into memory,
    and return as a tuple of numpy arrays

    Returns:
        Tuple of NumPy arrays: ``(x_train, y_train), (x_test, y_test)``

        **x_train**: uint8 NumPy array of grayscale image data with shapes
        ``(50000, 32, 32, 3)``, containing the training data. Pixel values range
        from 0 to 255.

        **y_train**: uint8 NumPy array of labels (integers in range 0-9)
        with shape ``(50000, 1)`` for the training data.

        **x_test**: uint8 NumPy array of grayscale image data with shapes
        (10000, 32, 32, 3), containing the test data. Pixel values range
        from 0 to 255.

        **y_test**: uint8 NumPy array of labels (integers in range 0-9)
        with shape ``(10000, 1)`` for the test data.


    """

    if dest_dir:
        dest_subdir = None

    path = download_verify_extract(
        url=DOWNLOAD_URL,
        file_hash=VERIFY_SHA1,
        show_progress=True,
        remove_root_dir=True,
        dest_dir=dest_dir,
        dest_subdir=dest_subdir,
        clean_dest_dir=clean_dest_dir,
        logger=logger
    )
    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000:i * 10000, :, :, :],
        y_train[(i - 1) * 10000:i * 10000]) = _load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = _load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if backend.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    x_test = x_test.astype(x_train.dtype)
    y_test = y_test.astype(y_train.dtype)

    return (x_train, y_train), (x_test, y_test)



def load_data_directory(
    dest_dir:str=None,
    dest_subdir='datasets/cifar10',
    logger:logging.Logger=None,
    clean_dest_dir=False
) -> str:
    """Download the dataset, extract all sample images to a directory,
    and return the path to the directory.

    Each sample type is extract to its corresponding subdirectory, e.g.:

    ~/.mltk/datasets/cifar10/airplane
    ~/.mltk/datasets/cifar10/automobile
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

                img = array_to_img(x, scale=False, dtype='uint8')
                img.save(sample_path)

    return dataset_dir




def _load_batch(fpath, label_key='labels'):
  """Internal utility for parsing CIFAR data.

  Arguments:
      fpath: path the file to parse.
      label_key: key for label data in the retrieve
          dictionary.

  Returns:
      A tuple `(data, labels)`.
  """
  with open(fpath, 'rb') as f:
    d = pickle.load(f, encoding='bytes')
    # decode utf8
    d_decoded = {}
    for k, v in d.items():
        d_decoded[k.decode('utf8')] = v
    d = d_decoded
  data = d['data']
  labels = d[label_key]

  data = data.reshape(data.shape[0], 3, 32, 32)
  return data, labels