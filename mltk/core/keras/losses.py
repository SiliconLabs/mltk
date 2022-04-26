"""
This contains custom loss functions that may be given to Keras, e.g:

 autoencoder.compile(loss=Correlation(), optimizer=Adam(lr=1e-3))
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.losses import (Reduction, Loss)
from tensorflow.keras.losses import MeanSquaredError # pylint: disable=unused-import


class Correlation(Loss):
    def __init__(self, 
                 reduction=Reduction.SUM_OVER_BATCH_SIZE,
                 name='corr'):
        super(Correlation, self).__init__(name=name, reduction=reduction)

    def call(self, y_true, y_pred):
        return corr_loss_func(y_true, y_pred)


class StructuralSimilarity(Loss):
    
    def __init__(self, 
                 reduction=Reduction.SUM_OVER_BATCH_SIZE,
                 name='ssim'):
        super(StructuralSimilarity, self).__init__(name=name, reduction=reduction)

    def call(self, y_true, y_pred):
        return ssim_loss_func(y_true, y_pred)


mse_loss_func = lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2)
corr_loss_func = lambda y_true, y_pred: -tfp.stats.correlation(y_true, y_pred, sample_axis=(-1,-2,-3), event_axis=None)


def ssim_loss_func(y_true, y_pred):
    # https://www.tensorflow.org/api_docs/python/tf/image/ssim
    # NOTE:
    # Input must be scaled between -1 to 1
    # Output activation must be -1 to 1 (e.g. tanh)
    return -tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))
