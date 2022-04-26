
from typing import List, Tuple, Callable, Union



from mltk.utils.python import as_list
from mltk.utils.string_formatting import format_units
from mltk.core.utils import get_mltk_logger

from .base_mixin import BaseMixin
from ..model_attributes import MltkModelAttributesDecorator
from ...tflite_model import TfliteModel


class TfliteModelMetadataEntry(object):
    """An entry to add to the generated .tflite model's metadata section"""
    def __init__(
            self, 
            tag:str, 
            serialize_callback:Callable=None,
            deserialize_callback:Callable=None,
            summary_callback:Callable=None
        ):
        self._tag = tag 
        self._serialize_callback = serialize_callback
        self._deserialize_callback = deserialize_callback
        self._summary_callback = summary_callback


    @property
    def tag(self) -> str:
        """The key used to find the metadata in the model .tflite flatbuffer's metadata section"""
        return self._tag


    def serialize(self) -> Tuple[str, bytes]:
        """Serialize the metadata into it byte string representation
        This device must be able to de-serialize the metadata for usage at run-time

        Typically, this API should serialize the data into a flatbuffer.

        Returns:
            None if no metadata should be added to the .tflite for this entry
            else tuple(tag, bytes)
        """
        if self._serialize_callback is None:
            raise Exception(f'MetadataEntry entry {self.tag} did not provide a serialize_callback')
        
        serialized_data = self._serialize_callback()
        if serialized_data is not None:
            return (self._tag, serialized_data)
        else:
            return None


    def deserialize(self, metadata_buffer: bytes):
        """De-serialize the metadata binary"""
        if self._deserialize_callback is not None:
            self._deserialize_callback(metadata_buffer)



    def summary(self) -> str:
        """Return a summary of this metadata"""
        if self._summary_callback is not None:
            return self._summary_callback()
        return None



@MltkModelAttributesDecorator()
class TfliteModelMetadataMixin(BaseMixin):
    """.tflite model Metadata mixin
    This mixin allows for reading/writing entries in the "metadata" section
    of a .tflite model flatbuffer file.
    """

    @property
    def tflite_metadata_entries(self) -> List[TfliteModelMetadataEntry]:
        """Return a list of registered metadata entries that will be included in the generated .tflite"""
        return self._attributes['tflite.model_metadata']
    

    def add_tflite_metadata_entry(self, entry: TfliteModelMetadataEntry):
        """Add an entry to the metadata that will be included in the generated .tflite"""
        self._attributes['tflite.model_metadata'].append(entry)


    def serialize_tflite_metadata(self) -> List[Tuple[str, bytes]]:
        """Serialize each metadata entry registered with this model

        This is invoked by the quantize_model() API when generated the model's .tflite file.
        
        Returns:
            list(tuple(<metedata tag>, <metadata serialized bytes>))
        """
        retval = []
        for entry in self.tflite_metadata_entries:
            tag_and_data = entry.serialize()
            if tag_and_data is None:
                continue
            retval.append(tag_and_data)

        return retval


    def deserialize_tflite_metadata(self, tflite_model:Union[bytes, str, TfliteModel]):
        """Load metadata from the given .tflite flatbuffer file
        The given tflite_model argument can be the .tflite's flatbuffer bytes,
        a file path to the .tflite file, or a TfliteModel object.
        """
        
        if isinstance(tflite_model, str):
            tflite_model = TfliteModel.load_flatbuffer_file(tflite_model)
        elif isinstance(tflite_model, (bytes,bytearray)):
            tflite_model = TfliteModel(tflite_model)
        elif not isinstance(tflite_model, TfliteModel):
            raise ValueError('tflite_model argument must be a .tflite file path, flatbuffer bytes, or TfliteModel object')

        tflite_metadata = tflite_model.get_all_metadata()
        for entry in self.tflite_metadata_entries:
            if entry.tag in tflite_metadata:
                metadata_buffer = tflite_metadata[entry.tag]
                del tflite_metadata[entry.tag]
                if metadata_buffer is not None:
                    get_mltk_logger().debug(f'De-serializing {entry.tag}')
                    entry.deserialize(metadata_buffer)
                else:
                    get_mltk_logger().debug(f'.tflite does not contain metadata: {entry.tag}')

        self._unparsed_metadata = tflite_metadata 
    

    def get_tflite_metadata_summary(self, include_tag=None, exclude_tag=None) -> str:
        """Generate a summary of the .tflite model's metadata"""
        include_tag = as_list(include_tag)
        exclude_tag = as_list(exclude_tag)
        s = ''
        for entry in self.tflite_metadata_entries:
            if include_tag and entry.tag not in include_tag:
                continue 
            if exclude_tag and entry.tag in exclude_tag:
                continue
            summary = entry.summary()
            if summary:
                s += f'{summary}\n'

        if hasattr(self, '_unparsed_metadata'):
            for key, value in self._unparsed_metadata.items():
                s += f'Unparsed metadata entry: {key}=<{format_units(len(value), add_space=False)}Bytes>\n'
                
        return s

    
    def _register_attributes(self):
        self._attributes.register('tflite.model_metadata', value=[], readonly=True)
