
from typing import Any, Callable



from .base_mixin import BaseMixin
from .tflite_model_metadata_mixin import TfliteModelMetadataEntry
from ..model_attributes import MltkModelAttributesDecorator
from ...tflite_model_parameters import TfliteModelParameters, TFLITE_METADATA_TAG


@MltkModelAttributesDecorator()
class TfliteModelParametersMixin(BaseMixin):
    """Model mixin for defining additional parameters to 
    include in the generate .tflite model's metadata"""

    @property
    def model_parameters(self) -> TfliteModelParameters:
        """Dictionary of model parameters to include in the generated .tflite"""
        return self._attributes['model.parameters']


    def set_model_parameter(self, key: str, value:Any, dtype:str=None):
        """Insert a custom entry into the tflite model paremeters"""
        self._attributes['model.parameters'].put(
            key=key, 
            value=value, 
            dtype=dtype
        )


    def add_model_parameter_populate_callback(self, callback: Callable):
        """Add a callback that populates the tflite model parameters
        The callback is invoked just before the model parameters are serialized

        NOTE: This is typically called by other MltkModel mixins
        """
        self._attributes['model.populate_callbacks'].append(callback)


    def populate_model_parameters(self):
        """Populate the model parameters.
        
        This is done by invoking callbacks that were registered by mixins like the AudioDatasetMixin
        which populates the TfliteModelParameters with the AudioFeatureGenerator settings

        This is automatically called by APIs like quantize_model() and update_model_parameters()
        """
        populate_callbacks = self._attributes['model.populate_callbacks']
        for callback in populate_callbacks:
            callback()


    def _populate_default_parameters(self):
        """Add the default parameters to the TfliteModelParameters
        This is called during self._serialize_parameters()
        """
        # Add the model name
        self.model_parameters['name'] = self.name
        # Add the model version
        self.model_parameters['version'] = self.version
        # Add the models classes if they're available
        try:
            self.model_parameters['classes'] = self.classes
        except:
            pass


    def _register_attributes(self):
        self._attributes.register('model.parameters', value=TfliteModelParameters(dict(
            name='',
            version=0,
            classes=[],
            hash='',
            date='',
            runtime_memory_size=0
        )))
        self._attributes.register('model.populate_callbacks', value=[self._populate_default_parameters], readonly=True)

        # We cannot call attributes while we're registering them
        # So we return a function that will be called after
        # all the attributes are registered
        def add_metadata_entry():
            self.add_tflite_metadata_entry(
                TfliteModelMetadataEntry(
                    tag=TFLITE_METADATA_TAG,
                    serialize_callback=self._serialize_parameters,
                    deserialize_callback=self._deserialize_parameters,
                    summary_callback=self._generate_summary
                )
            )
        
        return add_metadata_entry


    def _serialize_parameters(self) -> bytes:
        """Serialize the .tflite model parameters into flatbuffer
        
        This is called by TfliteModelMetadataMixin.serialize_tflite_metadata()
        """
        # First, call the populate callbacks
        # These callbacks are registered by mixins like the AudioDatasetMixin
        # which will populate the TfliteModelParameters with the AudioFeatureGenerator settings
        self.populate_model_parameters()

        return self.model_parameters.serialize()


    def _deserialize_parameters(self, metadata_buffer:bytes):
        """De-serialize the TfliteModelParameters from the flatbuffer bytes
        
        This is called by TfliteModelMetadataMixin.deserialize_tflite_metadata()
        """
        params = TfliteModelParameters.deserialize(metadata_buffer)
        # NOTE: Update the existing TfliteModelParameters rather than set so that we
        #       don't remove any new params not in the de-serialized model file
        self.model_parameters.update(params)
        if 'name' in params:
            self._attributes['name'] = params['name']
        if 'version' in params:
            self.version = params['version']
    
        try:
            self.classes = params['classes']
        except:
            pass

    def _generate_summary(self) -> str:
        """Generate a summary of the TfliteModelParameters"""
        parameters = self.model_parameters
        if len(parameters) == 0:
            populate_callbacks = self._attributes['model.populate_callbacks']
            for callback in populate_callbacks:
                callback()

        return parameters.summary()