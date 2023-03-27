from __future__ import annotations
import os
import warnings

from typing import List, Dict, Union, Iterator
from prettytable import PrettyTable

import numpy as np

# Disable the "DeprecationWarning" found in the flatbuffer package
warnings.filterwarnings("ignore", category=DeprecationWarning)

from . import tflite_schema as _tflite_schema_fb
from .tflite_schema import BuiltinOperator as TfliteOpCode # pylint: disable=unused-import
from .tflite_schema import flatbuffers

from .tflite_tensor import TfliteTensor
from .tflite_layer import TfliteLayer



TFLITE_FILE_IDENTIFIER = b"TFL3"




class TfliteModel:
    """Class to access a .tflite model flatbuffer's layers and tensors

    Refer to `schema_v3.fbs <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema_v3.fbs>`_
    for more details on the .tflite flatbuffer schema

    **Example Usage**

    .. highlight:: python
    .. code-block:: python

        from mltk.core import TfliteModel

        # Load you .tflite model file
        model = TfliteModel.load_flatbuffer_file('some/path/my_model.tflite')

        # Print a summary of the model
        print(tflite_model.summary())

        # Iterate through each layer of the model
        for layer in tflite_model.layers:
            # See TfliteLayer for additional info
            print(layer)


        # Update the model's description
        # This updates the .tflite's "description" field (which will be displayed in GUIs like https://netron.app)
        tflite_model.description = "My awesome model"
        print(f'New model description: {tflite_model.description}')

        # Save a new .tflite with the updated description
        tflite_model.save('some/path/my_new_model.tflite')

        # Add some metadata to the .tflite
        metadata = 'this is metadata'.encode('utf-8')
        tflite_model.add_metadata('my_metadata', metadata)

        # Retrieve all the metadata in the .tflite
        all_metadata = tflite_model.get_all_metadata()
        for key, data in all_metadata.items():
            print(f'{key}: length={len(data)} bytes')

        # Save a new .tflite with the updated metadata
        tflite_model.save('some/path/my_new_model.tflite')

        # You must have Tensorflow instance to perform this step
        # This will run inference with the given buffer and return
        # the results. The input_buffer can be:
        # - a single sample as a numpy array
        # - a numpy array of 1 or more samples
        # - A Python generator that returns (batch_x, batch_y)
        # inference_results = tflite_model.predict(..)
    """

    @staticmethod
    def load_flatbuffer_file(path: str, cwd=None) -> TfliteModel:
        """Load a .tflite flatbuffer file"""
        found_path = _existing_path(path, cwd=cwd)
        if found_path is None:
            raise FileNotFoundError(f'.tflite model file not found: {path}')

        with open(found_path, 'rb') as f:
            flatbuffer_data = f.read()

        return TfliteModel(flatbuffer_data=flatbuffer_data, path=found_path)


    def __init__(self, flatbuffer_data: bytes, path: str=None):
        self.path = path
        self._interpreter = None
        self._interpreter_batch_size = -1
        self._flatbuffer_data : bytes = flatbuffer_data
        self._model:_tflite_schema_fb.ModelT = None
        self._selected_model_subgraph_index = -1
        self._subgraphs: List[_TfliteSubgraph] = []
        self._load_model()

    @property
    def path(self) -> str:
        """Path to .tflite file
        Returns None if no path was specified.
        The path is normalized and backslashes are converted to forward slash
        """
        return None if self._path is None else os.path.normpath(self._path).replace('\\', '/')
    @path.setter
    def path(self, v: str):
        """Path to .tflite file"""
        if v is not None:
            v = v.replace('\\', '/')
        self._path = v

    @property
    def filename(self) -> str:
        """File name of associated .tflite model file
        Return None if not path is set"""
        if self._path:
            return os.path.basename(self._path)
        else:
            return None

    @property
    def name(self) -> str:
        """The name of the model which is the :py:attr:`~filename` without the ``.tflite`` extension or "my_model" if no path is set"""
        filename = self.filename
        if filename:
            if filename.endswith('.tflite'):
                filename = filename[:-len('.tflite')]
            return filename
        else:
            return 'my_model'

    @property
    def description(self) -> str:
        """Get/set model description

        .. note:: :py:func:`~save` must be called for changes to persist
        """
        return '' if self._model is None or not self._model.description else self._model.description.decode('utf-8')
    @description.setter
    def description(self, desc: str):
        if self._model is None:
            raise RuntimeError('Model not loaded')
        desc = desc or ''
        self._model.description = desc.encode('utf-8')
        self.regenerate_flatbuffer()

    @property
    def flatbuffer_data(self) -> bytes:
        """Flatbuffer binary data"""
        if self._flatbuffer_data is None:
            return None
        return bytes(self._flatbuffer_data)

    @property
    def flatbuffer_size(self) -> int:
        """Size of the model flatbuffer in bytes"""
        if self.flatbuffer_data is None:
            return 0
        return len(self.flatbuffer_data)

    def __len__(self) -> int:
        return self.flatbuffer_size

    @property
    def flatbuffer_model(self) -> _tflite_schema_fb.ModelT:
        """Flatbuffer schema Model object"""
        return self._model

    @property
    def flatbuffer_subgraph(self) -> _tflite_schema_fb.SubGraphT:
        """Flatbuffer schema model subgraph"""
        if self._model is None:
            return None
        return self._model.subgraphs[self._selected_model_subgraph_index]

    @property
    def selected_model_subgraph(self) -> int:
        """The index of the selected model subgraph.
        Other properties and APIs will return layers/tensors from the selected subgraph
        """
        return self._selected_model_subgraph_index
    @selected_model_subgraph.setter
    def selected_model_subgraph(self, v: int):
        if self._model is None:
            return -1
        if v < 0 or v >= self.n_subgraphs:
            raise ValueError('Invalid model subgraph index')
        self._selected_model_subgraph_index = v

    @property
    def n_subgraphs(self) -> int:
        """Return the number of model subgraphs"""
        if self._model is None:
            return 0
        return len(self._model.subgraphs)

    @property
    def n_inputs(self) -> int:
        """Return the number of model inputs"""
        if self.flatbuffer_subgraph is None:
            return 0
        return len(self.flatbuffer_subgraph.inputs)

    @property
    def inputs(self) -> List[TfliteTensor]:
        """List of all input tensors"""
        if self.flatbuffer_subgraph is None:
            return None
        retval = []
        for index in self.flatbuffer_subgraph.inputs:
            retval.append(self.get_tensor(index))

        return retval

    @property
    def n_outputs(self) -> int:
        """Return the number of model outputs"""
        if self.flatbuffer_subgraph is None:
            return 0
        return len(self.flatbuffer_subgraph.outputs)

    @property
    def outputs(self) -> List[TfliteTensor]:
        """List of all output tensors"""
        if self.flatbuffer_subgraph is None:
            return None
        retval = []
        for index in self.flatbuffer_subgraph.outputs:
            retval.append(self.get_tensor(index))

        return retval

    @property
    def layers(self) -> List[TfliteLayer]:
        """List of all model layers for the current subgraph"""
        if self._selected_model_subgraph_index == -1:
            return None
        return self._subgraphs[self._selected_model_subgraph_index].layers

    @property
    def tensors(self) -> List[TfliteTensor]:
        """List of all model tensors for the current subgraph"""
        if self._selected_model_subgraph_index == -1:
            return None
        return self._subgraphs[self._selected_model_subgraph_index].tensors


    def summary(self) -> str:
        """Generate a summary of the model"""
        if self._flatbuffer_data is None:
            return 'Not loaded'

        t = PrettyTable()
        t.field_names = [
            'Index',
            'OpCode',
            'Input(s)',
            'Output(s)',
            'Config'
        ]

        for i, layer in enumerate(self.layers):
            inputs = '\n'.join([x.shape_dtype_str(include_batch=False) for x in layer.inputs])
            outputs = '\n'.join([x.shape_dtype_str(include_batch=False) for x in layer.outputs])
            t.add_row([
                i,
                layer.opcode_str,
                inputs,
                outputs,
                f'{layer.options}'
            ])

        t.align = 'l'
        return t.get_string()


    def get_flatbuffer_subgraph(self, index:int=None) -> _tflite_schema_fb.SubGraphT:
        """Flatbuffer schema model subgraph at the given index

        If no index is given, then use the selected_model_subgraph
        """
        if self._model is None:
            raise RuntimeError('Model not loaded')
        index = index or self._selected_model_subgraph_index
        return self._model.subgraphs[index]


    def get_tensor(self, index : int) -> TfliteTensor:
        """Return a specific model tensor as a TfliteTensor """
        if self._model is None:
            raise RuntimeError('Model not loaded')
        subgraph = self._subgraphs[self._selected_model_subgraph_index]
        if index >= len(subgraph.tensors):
            raise IndexError(f'Index overflow ({index} >= {len(subgraph.tensors)})')
        return subgraph.tensors[index]


    def get_tensor_data(self, index : int) -> np.ndarray:
        """Return a specific model tensor as a np.ndarray """
        tensor = self.get_tensor(index=index)
        if tensor is None:
            return None
        return tensor.data


    def get_input_tensor(self, index: int = 0) -> TfliteTensor:
        """Return a model input tensor as a TfliteTensor"""
        if index >= self.n_inputs:
            raise IndexError(f'Index overflow ({index} >= {self.n_inputs})')
        tensor_index = self.flatbuffer_subgraph.inputs[index]
        return self.get_tensor(tensor_index)


    def get_input_data(self, index: int = 0) -> np.ndarray:
        """Return a model input as a np.ndarray"""
        if index >= self.n_inputs:
            raise IndexError(f'Index overflow ({index} >= {self.n_inputs})')
        tensor_index = self.flatbuffer_subgraph.inputs[index]
        return self.get_tensor_data(tensor_index)


    def get_output_tensor(self, index: int = 0) -> TfliteTensor:
        """Return a model output tensor as a TfliteTensor"""
        if index >= self.n_outputs:
            raise IndexError(f'Index overflow ({index} >= {self.n_outputs})')
        tensor_index = self.flatbuffer_subgraph.outputs[index]
        return self.get_tensor(tensor_index)


    def get_output_data(self, index: int = 0) -> np.ndarray:
        """Return a model output tensor as a np.ndarray"""
        if index >= self.n_outputs:
            raise IndexError(f'Index overflow ({index} >= {self.n_outputs})')
        tensor_index = self.flatbuffer_subgraph.outputs[index]
        return self.get_tensor_data(tensor_index)


    def get_all_metadata(self) -> Dict[str,bytes]:
        """Return all model metadata as a dictionary"""
        if self._model is None:
            raise RuntimeError('Model not loaded')
        retval = {}
        for metadata in self._model.metadata:
            name = metadata.name.decode("utf-8")
            buffer_index = metadata.buffer
            retval[name] = self._model.buffers[buffer_index].data.tobytes()

        return retval


    def get_metadata(self, tag : str) -> bytes:
        """Return model metadata with specified tag"""
        if self._model is None:
            raise RuntimeError('Model not loaded')
        metadata_value = None
        for metadata in self._model.metadata:
            if metadata.name.decode("utf-8") == tag:
                buffer_index = metadata.buffer
                metadata_value = self._model.buffers[buffer_index].data.tobytes()
                break

        return metadata_value


    def add_metadata(self, tag :str, value: bytes):
        """Set or add metadata to model

        .. Note::
            :func:`~tflite_model.TfliteModel.save` must be called for changes to persist

        Args:
            tag (str): The key to use to lookup the metadata
            value (bytes): The metadata value as a binary blob to add to the .tflite
        """
        if self._model is None:
            raise RuntimeError('Model not loaded')
        if not tag or not value:
            raise ValueError('Must provide valid tag and value arguments')


        buffer_field = _tflite_schema_fb.BufferT()
        buffer_field.data = np.frombuffer(value, dtype=np.uint8)

        add_buffer = False
        if not self._model.metadata:
            self._model.metadata = []
        else:
            # Check if metadata has already been add to the model.
            for meta in self._model.metadata:
                if meta.name.decode("utf-8") == tag:
                    add_buffer = True
                    self._model.buffers[meta.buffer] = buffer_field

        if not add_buffer:
            if not self._model.buffers:
                self._model.buffers = []
            self._model.buffers.append(buffer_field)
            # Creates a new metadata field.
            metadata_field = _tflite_schema_fb.MetadataT()
            metadata_field.name = tag.encode('utf-8')
            metadata_field.buffer = len(self._model.buffers) - 1
            self._model.metadata.append(metadata_field)

        self.regenerate_flatbuffer()


    def remove_metadata(self, tag: str) -> bool:
        """Remove model metadata with specified tag

        .. Note::
            :func:`~tflite_model.TfliteModel.save` must be called for changes to persist

        Args:
            tag (str): The key to use to lookup the metadata

        Return:
            True if the metadata was found and removed, False else

        """
        if self._model is None:
            raise RuntimeError('Model not loaded')

        if not self._model.metadata:
            return False

        removed_metadata = False
        for meta in self._model.metadata:
            if meta.name.decode("utf-8") == tag:
                removed_metadata = True
                self._model.metadata.remove(meta)
                self._model.buffers.pop(meta.buffer)
                self.regenerate_flatbuffer()
                break

        return removed_metadata


    def save(self, output_path: str = None):
        """Save flatbuffer data to file
        If output_path is specified then write to new file,
        otherwise overwrite existing file
        """
        output_path = output_path or self.path

        if not output_path:
            raise RuntimeError('No output path specified')

       # Re-generate the underlying flatbuffer
        self.regenerate_flatbuffer()

        # Create the model's output directory if necessary
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(output_path, 'wb') as f:
            f.write(self._flatbuffer_data)


    def regenerate_flatbuffer(self):
        """Re-generate the underlying flatbuffer based on  the information cached in the local ModelT instance

        .. Note::
            :func:`~tflite_model.TfliteModel.save` must be called for changes to persist
        """
        if self._model is None:
            raise RuntimeError('Model not loaded')
        b = flatbuffers.Builder(0)
        b.Finish(self._model.Pack(b), TFLITE_FILE_IDENTIFIER)
        self._flatbuffer_data = b.Output()


    def predict(
        self,
        x:Union[np.ndarray, Iterator, List[np.ndarray], Dict[int, np.ndarray]],
        y_dtype=None,
        **kwargs
    ) -> np.ndarray:
        """Invoke the TfLite interpreter with the given input sample and return the results

        If the model has a single input and output, the x data can one of:

        - A single sample as a numpy array
        - Iterable list of samples
        - Sample generator

        In this case, this API will manage converting
        the samples to the correct data type and adding the necessary batch dimension. The
        output value will either be list of model predictions or a single prediction corresponding to the input.

        If the model has multiple inputs and outputs, then the input data must be one of:

        - Python list of numpy arrays. One numpy array per model input. The numpy arrays must only contain the values for one sample.
          The input numpy arrays do NOT need to have the batch dimension. In this case, the output values will also not have the batch dimension.
        - Dictionary of one or more numpy arrays. The dictionary key should be an integer corresponding to the model input,
          and the value should be a numpy array. The input numpy arrays do NOT need to have the batch dimension.
          In this case, the output values will also not have the batch dimension.

        Args:
            x: The input samples(s) as a numpy array or data generator.
                If x is a numpy array then it must have the same shape as the model input
                or it must be a vector (i.e. batch) of samples having the same shape as the model input.
                The data type must either be the same as the model input's OR it must be a float32,
                in which case the input sample will automatically be quantized using the model input's
                quantizing scaler/zeropoint.
                If x is a generator, then each iteration must return a tuple: batch_x, batch_y
                batch_x must  be a vector (i.e. batch) of samples having the same shape as the model input
                batch_y is ignored.
            y_dtype: The return value's data type. By default, data type is None in which case the model output is directly returned.
                If y_dtype=np.float32 then the model output is de-quantized to float32 using the model's output
                quantization scaler/zeropoint (if necessary)

        Returns:
            Output of model inference, y. If x was a single sample, then y is a single result. Otherwise
            y is a vector (i.e. batch) of model results.
            If y_dtype is given, the y if automatically converted/de-quantized to the given dtype.
        """
        if self._flatbuffer_data is None:
            raise RuntimeError('Model not loaded')

        input0 = self.get_input_tensor(0)
        input0_shape = input0.shape

        if self.n_inputs > 1:
            if isinstance(x, (list,tuple)):
                if not all(isinstance(t, np.ndarray) for t in x):
                    raise ValueError('''
                        For multi-input models, the input data must be a list of numpy arrays
                    ''')
                x = {index: value for index, value in enumerate(x)}
            elif isinstance(x, dict):
                if not all(isinstance(k, int) and isinstance(v, np.ndarray) for k,v in x.items()):
                    raise ValueError('''
                        For multi-input models, the input data must be a dictionary of numpy arrays
                        with the keys corresponding to the model input index
                    ''')
            else:
                raise ValueError('''
                    For multi-input models, the input data must be a list of numpy arrays or
                    dictionary of numpy arrays with the keys corresponding to the model input index
                ''')

            # Set the input tensors
            has_batch_dim = True
            for input_index, x_i in x.items():
                if input_index == 0:
                    # Check if the input sample has the batch dimension
                    if len(x_i.shape) == len(input0_shape[1:]):
                        has_batch_dim = False
                        self._allocate_tflite_interpreter(
                            batch_size=1,
                            interpreter_kwargs=kwargs.get('interpreter_kwargs', None)
                        )
                    else:
                        self._allocate_tflite_interpreter(
                            batch_size=x_i.shape[0],
                            interpreter_kwargs=kwargs.get('interpreter_kwargs', None)
                        )

                # Add the batch_size=1 if the input sample doesn't have a batch dim
                if not has_batch_dim:
                    x_i = np.expand_dims(x_i, axis=0)

                # If the input sample isn't the same as the model input dtype,
                # then we need to manually convert it first
                # NOTE: If the model input type is float32 then
                #       quantization is done automatically inside the model
                x_i = self.quantize_to_input_dtype(x_i, input_index=input_index)
                self._interpreter.set_tensor(self.get_input_tensor(input_index).index, x_i)

            # Execute the model
            self._interpreter.invoke()

            # Get the model results
            y = []
            for i, outp in enumerate(self.outputs):
                y_i = self._interpreter.get_tensor(outp.index)

                # If the input doesn't have a batch dim
                # then remove the dim from the output
                if not has_batch_dim:
                    y_i = np.squeeze(y_i, axis=0)

                if y_dtype == np.float32:
                    # Convert the output data type to float32 if necessary
                    y_i = self.dequantize_output_to_float32(y_i, output_index=i)

                y.append(y_i)

            return y


        # This expects either
        # [n_samples, input_shape...]
        # OR
        # [input_shape ...]
        if isinstance(x, np.ndarray):
            is_single_sample = False
            if len(x.shape) == len(input0_shape[1:]):
                is_single_sample = True
                # Add the batch dimension if we were only given a single sample
                x = np.expand_dims(x, axis=0)
                self._allocate_tflite_interpreter(
                    batch_size=1,
                    interpreter_kwargs=kwargs.get('interpreter_kwargs', None)
                )
            else:
                self._allocate_tflite_interpreter(
                    batch_size=x.shape[0],
                    interpreter_kwargs=kwargs.get('interpreter_kwargs', None)
                )

            # If the input sample isn't the same as the model input dtype,
            # then we need to manually convert it first
            # NOTE: If the model input type is float32 then
            #       quantization is done automatically inside the model
            x = self.quantize_to_input_dtype(x)

            # If the last dimension of the model's input shape is 1,
            # and the input data is missing this dimension
            # then automatically expand the dimension
            if len(input0_shape) != len(x.shape) and input0_shape[-1] == 1:
                x = np.expand_dims(x, axis=-1)

            # Then set model input tensor
            self._interpreter.set_tensor(input0.index, x)
            # Execute the model
            self._interpreter.invoke()

            # Get the model results
            y = self._interpreter.get_tensor(self.get_output_tensor(0).index)

            # Convert the output data type to float32 if necessary
            # NOTE: If the model output type is float32 then
            #       de-quantization is done automatically inside the model

            if y_dtype == np.float32:
                y = self.dequantize_output_to_float32(y)

            # Remove the batch dimension if we were only given a single sample
            if is_single_sample:
                y = np.squeeze(y, axis=0)

            return y

        # Else if we were given a data generator
        else:
            n_samples = 0
            batch_results = []
            for batch in x:
                batch_x = batch if not isinstance(batch, tuple) else batch[0]

                self._allocate_tflite_interpreter(batch_size=batch_x.shape[0])

                # If the input sample isn't the same as the model input dtype,
                # then we need to manually convert it first
                batch_x = self.quantize_to_input_dtype(batch_x)

                # If the last dimension of the model's input shape is 1,
                # and the batch data is missing this dimension
                # then automatically expand the dimension
                if len(input0_shape) != len(batch_x.shape) and input0_shape[-1] == 1:
                    batch_x = np.expand_dims(batch_x, axis=-1)

                # The set model input tensor
                self._interpreter.set_tensor(input0.index, batch_x)
                # Execute the model
                self._interpreter.invoke()

                # Get the model results
                batch_y = self._interpreter.get_tensor(self.get_output_tensor(0).index)

                if y_dtype == np.float32:
                    # Convert the output data type to float32 if necessary
                    batch_y = self.dequantize_output_to_float32(batch_y)

                batch_results.append(batch_y)
                n_samples += len(batch_y)

                # If the generator specifies a "max_samples" property
                # then break out of the loop once the specified number of samples have been processed
                try:
                    if hasattr(x, 'max_samples') and x.max_samples > 0:
                        if n_samples >= x.max_samples:
                            break
                except:
                    pass

            if len(batch_results) == 0:
                raise RuntimeError('No batch samples where generated by the data given data generator')

            batch_size = batch_results[0].shape[0]
            output_shape = batch_results[0].shape[1:]

            if hasattr(x, 'max_samples') and x.max_samples > 0:
                n_samples = x.max_samples

            y = np.zeros((n_samples, *output_shape), dtype=batch_y.dtype)

            for batch_index, batch in enumerate(batch_results):
                for result_index, result in enumerate(batch):
                    index = batch_index * batch_size + result_index
                    if index >= n_samples:
                        break
                    y[index,:] = result

            return y


    def quantize_to_input_dtype(self, x:np.ndarray, input_index=0):
        """Quantize the input sample(s) to the model's input dtype (if necessary)"""

        input_tensor = self.get_input_tensor(input_index)

        if x.dtype == input_tensor.dtype:
            return x

        if x.dtype != np.float32:
            raise RuntimeError('The sample input must be float32 or the same dtype as the model input')

        # Convert from float32 to the model input data type
        x = (x / input_tensor.quantization.scale[0]) + input_tensor.quantization.zeropoint[0]
        return x.astype(input_tensor.dtype)


    def dequantize_output_to_float32(self, y:np.ndarray, output_index=0):
        """De-quantize the model output to float32 (if necessary)"""
        if y.dtype == np.float32:
            return y

        output_tensor = self.get_output_tensor(output_index)

        y = y.astype(np.float32)
        return (y - output_tensor.quantization.zeropoint[0]) * output_tensor.quantization.scale[0]



    def _allocate_tflite_interpreter(self, batch_size=1, interpreter_kwargs=None):
        if self._interpreter is None or self._interpreter_batch_size != batch_size:
            try:
                import tensorflow as tf
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(f'You must first install the "tensorflow" Python package to run inference, err: {e}') # pylint: disable=raise-missing-from

            interpreter_kwargs = interpreter_kwargs or {}
            self._interpreter_batch_size = batch_size
            self._interpreter = tf.lite.Interpreter(
                model_path=self._path,
                **interpreter_kwargs
            )

            input_indices = []
            for inp in self.inputs:
                input_indices.append(inp.index)
                new_input_shape = (batch_size, *inp.shape[1:])
                self._interpreter.resize_tensor_input(inp.index, new_input_shape)

            for outp in self.outputs:
                if outp.index in input_indices:
                    continue
                new_output_shape = (batch_size, *outp.shape[1:])
                self._interpreter.resize_tensor_input(outp.index, new_output_shape)

            self._interpreter.allocate_tensors()


    def _load_model(self):
        try:
            self._model = _tflite_schema_fb.ModelT.InitFromObj(_tflite_schema_fb.Model.GetRootAsModel(self._flatbuffer_data, 0))
            subgraph_count = len(self._model.subgraphs)
        except Exception as e:
            raise RuntimeError( # pylint: disable=raise-missing-from
                'Failed to load .tflite model flatbuffer.\n'
                'Ensure you have provided a valid .tflite model (i.e. ensure the binary data has not been corrupted)\n'
                f'Error details: {e}'
            )

        schema_version = self._model.version
        if schema_version != 3:
            raise RuntimeError('TF-Lite schema v3 is only supported')

        if self._selected_model_subgraph_index == -1 or self._selected_model_subgraph_index >= subgraph_count:
            self._selected_model_subgraph_index = 0

        self._subgraphs = []
        for fb_subgraph in self._model.subgraphs:
            subgraph = _TfliteSubgraph()
            self._subgraphs.append(subgraph)
            for i, fb_tensor in enumerate(fb_subgraph.tensors):
                tensor = TfliteTensor(i, self, fb_tensor)
                subgraph.tensors.append(tensor)
            for i, operator in enumerate(fb_subgraph.operators):
                layer = TfliteLayer.from_flatbuffer(i, self, operator)
                subgraph.layers.append(layer)



def _existing_path(path: str, cwd=None):
    if path is None:
        return None

    if cwd:
        found_path = f'{cwd}/{path}'
        if os.path.exists(found_path):
            return found_path

    if os.path.exists(path):
        return path

    return None

class _TfliteSubgraph:
    def __init__(self):
        self.layers: List[TfliteLayer] = []
        self.tensors: List[TfliteTensor] = []

