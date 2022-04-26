


from .evaluate_mixin import EvaluateMixin
from ..model_attributes import MltkModelAttributesDecorator



@MltkModelAttributesDecorator()
class EvaluateClassifierMixin(EvaluateMixin):
    """Provides evaluation properties and methods to the base :py:class:`~MltkModel`
    
    .. note:: This mixin is specific to "classification" models

    Refer to the `Model Evaluation <https://siliconlabs.github.io/mltk/docs/guides/model_evaluation.html>`_ guide for more details.
    """

    @property
    def eval_shuffle(self) -> bool:
        """Shuffle data during evaluation
        Default: `False`
        """
        return self._attributes.get_value('eval.shuffle', default=False)
    @eval_shuffle.setter
    def eval_shuffle(self, v: bool):
        self._attributes['eval.shuffle'] = v 


    @property
    def eval_augment(self) -> bool:
        """Enable random augmentations during evaluation
        Default: `False`
        NOTE: This is only used if the DataGeneratorDatasetMixin or sub-class is used by the MltkModel
       
        """
        return self._attributes.get_value('eval.augment', default=False)
    @eval_augment.setter 
    def eval_augment(self, v: bool):
        self._attributes['eval.augment'] = v 


    @property
    def eval_max_samples_per_class(self) -> int:
        """The maximum number of samples for a given class to use during evaluation. If `-1` then use all available samples
        Default: `-1`
        """
        return self._attributes.get_value('eval.max_samples_per_class', default=-1)
    @eval_max_samples_per_class.setter 
    def eval_max_samples_per_class(self, v: int):
        self._attributes['eval.max_samples_per_class'] = v 



    def _register_attributes(self):
        self._attributes.register('eval.shuffle', dtype=bool)
        self._attributes.register('eval.augment', dtype=bool)
        self._attributes.register('eval.max_samples_per_class', dtype=int)