"""Utilities for processing image data"""

from typing import Union, Tuple
import os
import tensorflow as tf
import numpy as np


def read_image_file(
    path:Union[str,np.ndarray,tf.Tensor],
    target_channels=0,
    return_numpy=False
) -> Union[np.ndarray,tf.Tensor]:
    """Reads and decodes an image file.

    Args:
        path: Path to image file as a python string, numpy string, or tensorflow string
        target_channels: Number of channels to return image as, if 0 then use native channels
        return_numpy: If true then return numpy array, else return TF tensor

    Returns:
        Image data as numpy array or TF tensor
    """
    if target_channels is None:
        target_channels = 0

    raw = tf.io.read_file(path)
    sample = tf.io.decode_image(
        raw,
        expand_animations=True,
        channels=target_channels,
        dtype=tf.uint8
    )
    if return_numpy:
        return sample.numpy()

    return sample


def write_image_file(
    path:str,
    sample:Union[np.ndarray,tf.Tensor],
    auto_scale=True,
    batch_size:int=None
) -> Union[str,tf.Tensor]:
    """Write image data to a file

    Args:
        path: File path to save image
            If this is does NOT end with .jpg, then the path is assumed to be a directory.
            In this case, the image path is generated as: <path>/<timestamp>.jpg
        sample: Image data to write, if the data type is:
            - ``int8`` then it is converted to uint8
            - ``float32`` and ``auto_scale=True`` then the image is automatically scaled to 0-255
        auto_scale: Automatically scale the image data to 0-255 if it is ``float32``
        batch_size: This allows for using this function within a tf.keras.layers.Lambda layer
            If used, this will write each image in the given batch
    Returns:
        Path to written file. If this is executing in a non-eager TF function
        then the path is a TF Tensor, otherwise it is a Python string
    """
    if isinstance(sample, np.ndarray):
        sample = tf.convert_to_tensor(sample)


    if batch_size and len(sample.shape) == 4:
        partitions = tf.range(batch_size)
        partitioned = tf.dynamic_partition(sample, partitions, batch_size)
        for img in partitioned:
            img = tf.squeeze(img, axis=0)
            write_image_file(path, img, auto_scale=auto_scale)
        return

    if len(sample.shape) == 2:
        sample = tf.expand_dims(sample, axis=-1)

    if sample.shape[-1] == 1:
        format = 'grayscale'
    else:
        format = 'rgb'

    if sample.dtype == tf.int8:
        sample = tf.cast(sample, tf.int32)
        sample = sample + 128
        sample = tf.clip_by_value(sample, 0, 255)

    elif sample.dtype == tf.float32:
        if auto_scale:
            min_val = tf.math.reduce_min(sample)
            max_val = tf.math.reduce_max(sample)
            sample = (sample * 255.0 / ((max_val - min_val) + 1e-6))

        sample = tf.clip_by_value(sample, 0.0, 255.0)

    sample = tf.cast(sample, tf.uint8)

    path = tf.strings.join((os.path.abspath(path),))
    if not tf.strings.regex_full_match(path, r'.*\.jpg'):
        ts = tf.timestamp() * 1000
        fn = tf.strings.format('{}.jpg', ts)
        path = tf.strings.join((path, fn), separator=os.path.sep)

    img = tf.image.encode_jpeg(sample, quality=100, format=format)
    tf.io.write_file(path, img)

    if tf.executing_eagerly() and isinstance(path, tf.Tensor):
        path = path.numpy().decode('utf-8').replace('\\', '/')

    return path
