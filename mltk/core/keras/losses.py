"""
This contains custom loss functions that may be given to Keras, e.g:

 autoencoder.compile(loss=Correlation(), optimizer=Adam(lr=1e-3))
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.losses import (Reduction, Loss)
from tensorflow.keras.losses import MeanSquaredError # pylint: disable=unused-import
from tensorflow.keras.losses import MeanAbsoluteError # pylint: disable=unused-import


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
mae_loss_func = lambda y_true, y_pred: np.mean(np.abs(y_pred - y_true), axis=-1)

def ssim_loss_func(y_true, y_pred):
    # https://www.tensorflow.org/api_docs/python/tf/image/ssim
    # NOTE:
    # Input must be scaled between -1 to 1
    # Output activation must be -1 to 1 (e.g. tanh)
    return -tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))



class ContrastiveLoss(Loss):
    def __init__(
        self, 
        margin=1, 
        reduction=Reduction.AUTO,
        name='contrastive_loss'
    ):
        """Calculates the contrastive loss.

        Contrastive loss = mean( (1-true_value) * square(prediction) +
        true_value * square( max(margin-prediction, 0) ))

        Arguments:

        margin: Integer, defines the baseline for distance for which pairs
                should be classified as dissimilar. - (default is 1).

        Returns:
            A tensor containing contrastive loss as floating point value.
        """
        super(ContrastiveLoss, self).__init__(
            name=name,
            reduction=reduction
        )
        self.margin = margin

    def call(self, y_true, y_pred):
        """
        Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.
        """
        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(self.margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    def get_config(self):
        """Returns the config dictionary for a `Loss` instance."""
        config =  super(ContrastiveLoss, self).get_config()
        config['margin'] = self.margin
        return config


