import logging

import numpy as np
from mltk.utils.logger import get_logger as _get_logger


class ArchiveFileNotFoundError(FileNotFoundError):
    '''File not found in model archive exception'''



def get_mltk_logger() -> logging.Logger:
    """Return the MLTK logger
    If not logger has been previously set, then create a default logger
    """
    if 'logger' not in globals():
        logger = _get_logger('mltk', console=True)
        set_mltk_logger(logger)

    return globals()['logger'] 


def set_mltk_logger(logger: logging.Logger):
    """Set the MLTK logger"""
    globals()['logger']  = logger



def convert_y_to_labels(y: np.ndarray) -> np.ndarray:
    """Convert a 1-hot encoded list of samples to a 1D list of class labels"""
    if len(y.shape) == 2 and y.shape[1] == 1:
        return np.squeeze(y, axis=1)

    labels = np.zeros(y.shape[0], dtype=int)
    for i in range(len(labels)): # pylint: disable=consider-using-enumerate
        labels[i] = np.where(y[i,:] == 1)[0]
    return labels


