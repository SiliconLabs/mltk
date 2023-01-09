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

    logger = globals()['logger']

    # Try to add TF logs to the MLTK log file
    try:
        tf_logger = logging.getLogger('tensorflow')
        mltk_file_handler = logger.file_handler
        if mltk_file_handler:
            already_added = False
            for h in tf_logger.handlers:
                if h == mltk_file_handler:
                    already_added = True
                    break

            if not already_added:
                tf_logger.addHandler(mltk_file_handler)
    except:
        pass

    return logger


def set_mltk_logger(logger: logging.Logger):
    """Set the MLTK logger"""
    globals()['logger']  = logger



def convert_y_to_labels(y: np.ndarray) -> np.ndarray:
    """Convert a 1-hot encoded list of samples to a 1D list of class labels"""
    if len(y.shape) == 2 and y.shape[1] == 1:
        return np.squeeze(y, axis=1)

    labels = np.zeros(y.shape[0], dtype=int)
    for i in range(len(labels)): # pylint: disable=consider-using-enumerate
        labels[i] = np.argmax(y[i])
    return labels


