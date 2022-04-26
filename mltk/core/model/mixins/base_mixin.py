
from ..model_attributes import MltkModelAttributes




class BaseMixin(object):
    _attributes: MltkModelAttributes = None

    def _register_attributes(self):
        pass  

