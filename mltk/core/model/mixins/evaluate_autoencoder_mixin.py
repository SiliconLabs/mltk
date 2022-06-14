
from typing import Callable, List


from .evaluate_classifier_mixin import EvaluateClassifierMixin
from ..model_attributes import MltkModelAttributesDecorator, CallableType



@MltkModelAttributesDecorator()
class EvaluateAutoEncoderMixin(EvaluateClassifierMixin):
    """Provides evaluation properties and methods to the base :py:class:`~MltkModel`
    
    .. note:: This mixin is specific to "auto-encoder" models

    Refer to the `Model Evaluation <https://siliconlabs.github.io/mltk/docs/guides/model_evaluation.html>`_ guide for more details.
    """


    @property
    def scoring_function(self) -> Callable:
        """The auto-encoder scoring function to use during evaluation

        If `None`, then use the `mltk_model.loss` function

        Default: `None`
        """
        return self._attributes.get_value('eval_autoencoder.scoring_function', default=None)
    @scoring_function.setter 
    def scoring_function(self, v: Callable):
        self._attributes['eval_autoencoder.scoring_function'] = v
    

    @property
    def eval_classes(self) -> List[str]:
        """List if classes to use for evaluation. 
        The first element should be considered the 'normal' class, every other class is considered abnormal and compared independently.
        This is used if the `--classes` argument is not supplied to the `eval` command.

        Default: `[normal, abnormal]`
        """
        return self._attributes.get_value('eval_autoencoder.classes', default=['normal', 'abnormal'] )
    @eval_classes.setter
    def eval_classes(self, v: List[str]):
        self._attributes['eval_autoencoder.classes'] = v


    def get_scoring_function(self) -> Callable:
        """Return the scoring function used during evaluation"""
        from mltk.core.keras.losses import (
            Correlation, 
            MeanSquaredError, 
            MeanAbsoluteError,
            mse_loss_func, 
            corr_loss_func,
            mae_loss_func
        )

        if self.scoring_function is not None:
            return self.scoring_function

        loss = self.loss 
        if loss in ('mse', 'mean_squared_error') or isinstance(loss, MeanSquaredError):
            return mse_loss_func
        elif loss in ('mae', 'mean_absolute_error') or isinstance(loss, MeanAbsoluteError):
            return mae_loss_func
        elif loss in ('corr', 'correlation') or isinstance(loss, Correlation):
            return corr_loss_func
        else:
            raise Exception('Only model loss functions: "mse", "mae", "corr" are supported by default.\n'
                           'You must specify mltk_model.scoring_function for your model')


    def _register_attributes(self):
        self._attributes.register('eval_autoencoder.scoring_function', dtype=CallableType)
        self._attributes.register('eval_autoencoder.classes', dtype=(list,tuple))