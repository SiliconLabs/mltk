

from typing import Callable
from .base_mixin import BaseMixin
from ..model_attributes import MltkModelAttributesDecorator, CallableType


@MltkModelAttributesDecorator()
class EvaluateMixin(BaseMixin):
    """Provides generic evaluation properties and methods to the base :py:class:`~MltkModel`
    
    Refer to the `Model Evaluation <https://siliconlabs.github.io/mltk/docs/guides/model_evaluation.html>`_ guide for more details.
    """

    @property
    def eval_steps_per_epoch(self) -> int:
        """Total number of steps (batches of samples) before declaring the prediction round finished. 
        Ignored with the default value of None. If x is a tf.data dataset and steps is None, 
        predict will run until the input dataset is exhausted.
        """
        return self._attributes.get_value('eval.steps_per_epoch', default=None)
    @eval_steps_per_epoch.setter 
    def eval_steps_per_epoch(self, v: int):
        self._attributes['eval.steps_per_epoch'] = v


    @property
    def eval_custom_function(self) -> Callable:
        """Custom evaluation callback
        
        This is invoked during the :py:func:`mltk.core.evaluate_model` API.

        The given function should have the following signature:

        .. highlight:: python
        .. code-block:: python

           my_custom_eval_function(my_model:MyModel, built_model: Union[KerasModel, TfliteModel]) -> EvaluationResults:
               results = EvaluationResults(name=my_model.name)

               if isinstance(built_model, KerasModel):
                   results['overall_accuracy] = calculate_accuracy(built_model)
               return results
        
        """
        return self._attributes.get_value('eval.custom_function', default=None)
    
    @eval_custom_function.setter
    def eval_custom_function(self, func:Callable):
        self._attributes['eval.custom_function'] = func



    def _register_attributes(self):
        self._attributes.register('eval.steps_per_epoch', dtype=int)
        self._attributes.register('eval.custom_function', dtype=CallableType)
