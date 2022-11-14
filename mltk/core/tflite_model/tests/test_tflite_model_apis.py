
import os
import pytest
import numpy as np



from mltk.core import (
    TfliteModel, 
    TfliteTensor, 
    TfliteOpCode,
    TfliteConv2dLayer,
    TfliteDepthwiseConv2dLayer,
    TfliteFullyConnectedLayer,
    TflitePooling2dLayer
)
from mltk.core.tflite_model.tflite_layer import TfliteConv2DLayerOptions
from mltk.utils.test_helper.data import IMAGE_CLASSIFICATION_TFLITE_PATH
from mltk.utils.path import create_tempdir


def test_load_flatbuffer_file():
    TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)

def test_load_flatbuffer_bytes():
    with open(IMAGE_CLASSIFICATION_TFLITE_PATH, 'rb') as fp:
        tflite_bytes = fp.read()
    TfliteModel(tflite_bytes, path=IMAGE_CLASSIFICATION_TFLITE_PATH)


def test_path_api():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)
    assert tflite_model.path == IMAGE_CLASSIFICATION_TFLITE_PATH
    tflite_model.path = 'new_path.tflite'
    assert tflite_model.path == 'new_path.tflite'

def test_filename_api():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)
    assert tflite_model.filename == os.path.basename(IMAGE_CLASSIFICATION_TFLITE_PATH)

def test_description_api():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)
    tflite_model.description = 'test description'
    assert tflite_model.description == 'test description'


def test_description_with_save_api():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)
    tflite_model.description = 'test description'
    assert tflite_model.description == 'test description'

    tmp_tflite_path = f'{create_tempdir()}/cifar10_resnet_v1.tflite'
    tflite_model.save(tmp_tflite_path)

    try:
        tflite_model = TfliteModel.load_flatbuffer_file(tmp_tflite_path)
        assert tflite_model.description == 'test description'
    finally:
        os.remove(tmp_tflite_path)


def test_flatbuffer_data_api():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)
    with open(IMAGE_CLASSIFICATION_TFLITE_PATH, 'rb') as fp:
        tflite_bytes = fp.read()

    assert tflite_model.flatbuffer_data == tflite_bytes

def test_flatbuffer_size_api():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)
    assert tflite_model.flatbuffer_size == os.path.getsize(IMAGE_CLASSIFICATION_TFLITE_PATH)

def test_n_inputs_api():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)
    assert tflite_model.n_inputs == 1

def test_n_outputs_api():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)
    assert tflite_model.n_outputs == 1 

def test_inputs_api():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)
    inputs = tflite_model.inputs

    assert len(inputs) == 1

    input_0 = inputs[0]
    assert f'{input_0}' == 'input_1_int8, dtype:int8, shape:1x32x32x3'
    assert input_0.index == 0
    assert input_0.name == 'input_1_int8'
    assert input_0.dtype == np.int8
    assert input_0.dtype_str == 'int8'
    assert input_0.shape == (1, 32, 32, 3)
    assert f'{input_0.shape}' == '1x32x32x3'
    assert input_0.shape.flat_size == 1*32*32*3
    assert input_0.shape_dtype_str() == '32x32x3 (int8)'
    assert input_0.shape_dtype_str(include_batch=True) == '1x32x32x3 (int8)'
    assert isinstance(input_0.data, np.ndarray)
    assert input_0.data.dtype == np.int8
    assert input_0.data.shape == (1, 32, 32, 3)
    assert input_0.quantization.n_channels == 1
    assert input_0.quantization.quantization_dimension == 0
    assert len(input_0.quantization.scale) == 1
    assert input_0.quantization.scale[0] == 1.0
    assert len(input_0.quantization.zeropoint) == 1
    assert input_0.quantization.zeropoint[0] == -128

def test_outputs_api():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)
    outputs = tflite_model.outputs

    assert len(outputs) == 1

    output_0 = outputs[0]
    assert f'{output_0}' == 'Identity_int8, dtype:int8, shape:1x10'
    assert output_0.index == 37
    assert output_0.name == 'Identity_int8'
    assert output_0.dtype == np.int8
    assert output_0.dtype_str == 'int8'
    assert output_0.shape == (1, 10)
    assert f'{output_0.shape}' == '1x10'
    assert output_0.shape.flat_size == 1*10
    assert output_0.shape_dtype_str() == '10 (int8)'
    assert output_0.shape_dtype_str(include_batch=True) == '1x10 (int8)'
    assert isinstance(output_0.data, np.ndarray)
    assert output_0.data.dtype == np.int8
    assert output_0.data.shape == (1, 10)
    assert output_0.quantization.n_channels == 1
    assert output_0.quantization.quantization_dimension == 0
    assert len(output_0.quantization.scale) == 1
    assert output_0.quantization.scale[0] == 0.00390625
    assert len(output_0.quantization.zeropoint) == 1
    assert output_0.quantization.zeropoint[0] == -128


def test_layers_api():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)
    layers = tflite_model.layers
    assert len(layers) == 16

def test_layer_api():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)
    layer_0 = tflite_model.layers[0]
    assert isinstance(layer_0, TfliteConv2dLayer)
    assert layer_0.index == 0
    assert layer_0.name == 'op0-conv_2d'
    assert layer_0.opcode == TfliteOpCode.CONV_2D
    assert layer_0.opcode_str == 'conv_2d'
    options = layer_0.options 
    assert isinstance(options, TfliteConv2DLayerOptions)
    assert len(layer_0.inputs) == 3
    assert layer_0.n_inputs == 3
    assert len(layer_0.outputs) == 1
    assert layer_0.n_outputs == 1
    input_0_tensor = layer_0.get_input_tensor(0)
    assert input_0_tensor.shape == (1,32,32,3)
    input_0 = layer_0.get_input_data(0)
    assert input_0.shape == (1,32,32,3)
    output_0_tensor = layer_0.get_output_tensor(0)
    assert output_0_tensor.shape == (1,32,32,16)
    output_0 = layer_0.get_output_data(0)
    assert output_0.shape == (1,32,32,16)

def test_summary_api():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)
    summary = tflite_model.summary()
    assert summary is not None


def test_get_tensor_api():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)
    t = tflite_model.get_tensor(0)
    assert isinstance(t, TfliteTensor)
    assert t.name == 'input_1_int8'

def test_get_tensor_bad_index_api():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)
    with pytest.raises(IndexError):
        tflite_model.get_tensor(99)

def test_get_tensor_data_api():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)
    t = tflite_model.get_tensor_data(0)
    assert isinstance(t, np.ndarray)

def test_get_input_tensor_api():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)
    t = tflite_model.get_input_tensor(0)
    assert isinstance(t, TfliteTensor)

def test_get_input_data_api():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)
    t = tflite_model.get_input_data(0)
    assert isinstance(t, np.ndarray)

def test_get_output_tensor_api():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)
    t = tflite_model.get_output_tensor(0)
    assert isinstance(t, TfliteTensor)

def test_get_output_data_api():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)
    t = tflite_model.get_output_data(0)
    assert isinstance(t, np.ndarray)

def test_all_metadata_api():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)
    metadata = tflite_model.get_all_metadata()
    assert len(metadata['min_runtime_version']) == 16

def test_get_metadata_api():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)

    metadata = tflite_model.get_metadata('min_runtime_version')
    assert len(metadata) == 16

def test_add_metadata_api():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)

    test_data = 'test data'.encode('utf-8')
    tflite_model.add_metadata('test', test_data)
    metadata = tflite_model.get_metadata('test')
    assert metadata == test_data

def test_add_metadata_with_save_api():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)

    test_data = 'test data'.encode('utf-8')
    tflite_model.add_metadata('test', test_data)
    metadata = tflite_model.get_metadata('test')
    assert metadata == test_data

    tmp_tflite_path = f'{create_tempdir()}/cifar10_resnet_v1.tflite'
    tflite_model.save(tmp_tflite_path)

    try:
        tflite_model = TfliteModel.load_flatbuffer_file(tmp_tflite_path)
        metadata = tflite_model.get_metadata('test')
        assert metadata == test_data
    finally:
        os.remove(tmp_tflite_path)


def test_remove_metadata_api():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)

    test_data = 'test data'.encode('utf-8')
    tflite_model.add_metadata('test', test_data)
    metadata = tflite_model.get_metadata('test')
    assert metadata == test_data

    assert tflite_model.remove_metadata('test')

def test_remove_metadata_with_save_api():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)

    test_data = 'test data'.encode('utf-8')
    tflite_model.add_metadata('test', test_data)
    metadata = tflite_model.get_metadata('test')
    assert metadata == test_data

    tmp_tflite_path = f'{create_tempdir()}/cifar10_resnet_v1.tflite'
    tflite_model.save(tmp_tflite_path)

    try:
        tflite_model = TfliteModel.load_flatbuffer_file(tmp_tflite_path)
        metadata = tflite_model.get_metadata('test')
        assert metadata is not None

        assert tflite_model.remove_metadata('test')

        tmp_tflite_path = f'{create_tempdir()}/cifar10_resnet_v1.tflite'
        tflite_model.save(tmp_tflite_path)

        tflite_model = TfliteModel.load_flatbuffer_file(tmp_tflite_path)
        metadata = tflite_model.get_metadata('test')
        assert metadata is None

    finally:
        os.remove(tmp_tflite_path)

def test_replace_metadata_api():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)

    test_data = 'test data'.encode('utf-8')
    tflite_model.add_metadata('test', test_data)
    metadata = tflite_model.get_metadata('test')
    assert metadata == test_data

    replaced_data = 'replaced data'.encode('utf-8')
    tflite_model.add_metadata('test', replaced_data)
    metadata = tflite_model.get_metadata('test')
    assert metadata == replaced_data


def test_replace_metadata_with_save_api():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)

    test_data = 'test data'.encode('utf-8')
    tflite_model.add_metadata('test', test_data)
    metadata = tflite_model.get_metadata('test')
    assert metadata == test_data

    replaced_data = 'replaced data'.encode('utf-8')
    tflite_model.add_metadata('test', replaced_data)
    metadata = tflite_model.get_metadata('test')
    assert metadata == replaced_data

    tmp_tflite_path = f'{create_tempdir()}/cifar10_resnet_v1.tflite'
    tflite_model.save(tmp_tflite_path)

    try:
        tflite_model = TfliteModel.load_flatbuffer_file(tmp_tflite_path)
        metadata = tflite_model.get_metadata('test')
        assert metadata == replaced_data
    finally:
        os.remove(tmp_tflite_path)


def test_predict_single_sample():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)

    x = np.zeros((32,32,3), dtype=np.int8)
    try:
        y = tflite_model.predict(x)
    except ModuleNotFoundError as e:
        print(f'WARN: Failed to import tensorflow, err: {e}')
        return 

    assert y.dtype == np.int8
    assert len(y) == 10

def test_predict_multi_sample():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)

    x = np.zeros((5, 32,32,3), dtype=np.int8)
    try:
        y = tflite_model.predict(x)
    except ModuleNotFoundError as e:
        print(f'WARN: Failed to import tensorflow, err: {e}')
        return 

    assert y.dtype == np.int8
    assert len(y.shape) == 2
    assert y.shape[0] == 5
    assert y.shape[1] == 10


def test_predict_generator():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)

    class _Iterator:
        def __init__(self):
            self.i = -1
        
        def __iter__(self):
            self.i = 0
            return self 

        def __next__(self):
            if self.i >= 3:
                raise StopIteration
            self.i += 1
            x = np.zeros((3, 32,32,3), dtype=np.int8)
            y = np.zeros((3,3), dtype=np.int8)
            return x, y

    try:
        y = tflite_model.predict(_Iterator())
    except ModuleNotFoundError as e:
        print(f'WARN: Failed to import tensorflow, err: {e}')
        return 

    assert y.dtype == np.int8
    assert len(y.shape) == 2
    assert y.shape[0] == 9
    assert y.shape[1] == 10 


def test_predict_generator_float32_input():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)

    class _Iterator:
        def __init__(self):
            self.i = -1
        
        def __iter__(self):
            self.i = 0
            return self 

        def __next__(self):
            if self.i >= 3:
                raise StopIteration
            self.i += 1
            x = np.zeros((3, 32,32,3), dtype=np.float32)
            y = np.zeros((3,3), dtype=np.int8)
            return x, y

    try:
        y = tflite_model.predict(_Iterator())
    except ModuleNotFoundError as e:
        print(f'WARN: Failed to import tensorflow, err: {e}')
        return 

    assert y.dtype == np.int8
    assert len(y.shape) == 2
    assert y.shape[0] == 9
    assert y.shape[1] == 10 


def test_predict_generator_float32_output():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)

    class _Iterator:
        def __init__(self):
            self.i = -1
        
        def __iter__(self):
            self.i = 0
            return self 

        def __next__(self):
            if self.i >= 3:
                raise StopIteration
            self.i += 1
            x = np.zeros((3, 32,32,3), dtype=np.int8)
            y = np.zeros((3,3), dtype=np.float32)
            return x, y

    try:
        y = tflite_model.predict(_Iterator(), y_dtype=np.float32)
    except ModuleNotFoundError as e:
        print(f'WARN: Failed to import tensorflow, err: {e}')
        return 

    assert y.dtype == np.float32
    assert len(y.shape) == 2
    assert y.shape[0] == 9
    assert y.shape[1] == 10 


def test_predict_float32_input():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)

    x = np.zeros((32,32,3), dtype=np.float32)
    try:
        y = tflite_model.predict(x)
    except ModuleNotFoundError as e:
        print(f'WARN: Failed to import tensorflow, err: {e}')
        return 

    assert y.dtype == np.int8
    assert len(y) == 10

def test_predict_float32_output():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)

    x = np.zeros((32,32,3), dtype=np.int8)
    try:
        y = tflite_model.predict(x, y_dtype=np.float32)
    except ModuleNotFoundError as e:
        print(f'WARN: Failed to import tensorflow, err: {e}')
        return 

    assert y.dtype == np.float32
    assert len(y) == 10

def test_predict_float32_input_and_output():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)

    x = np.zeros((32,32,3), dtype=np.float32)
    try:
        y = tflite_model.predict(x, y_dtype=np.float32)
    except ModuleNotFoundError as e:
        print(f'WARN: Failed to import tensorflow, err: {e}')
        return 

    assert y.dtype == np.float32
    assert len(y) == 10


def test_predict_different_batch_sizes():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)

    x1 = np.zeros((32,32,3), dtype=np.int8)
    x2 = np.zeros((5,32,32,3), dtype=np.int8)
    try:
        y1 = tflite_model.predict(x1)
        y2 = tflite_model.predict(x2)
    except ModuleNotFoundError as e:
        print(f'WARN: Failed to import tensorflow, err: {e}')
        return 

    assert y1.dtype == np.int8
    assert len(y1) == 10
    assert len(y2.shape) == 2
    assert y2.shape[0] == 5
    assert y2.shape[1] == 10


def test_load_corrupt_tflite():
    bogus_tflite = b'\x12\x34\x56\x78'

    try:
        TfliteModel(bogus_tflite)
    except Exception as e:
        assert isinstance(e, RuntimeError)
        return 

    assert False, 'Failed to detect corrupt tflite file'


def test_set_tensor_data():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)

    conv2d_layer:TfliteConv2dLayer = tflite_model.layers[0]
    older_filters_data = conv2d_layer.filters_data
    new_filters_data = np.random.randint(0, 127, size=older_filters_data.shape, dtype=older_filters_data.dtype)

    conv2d_layer.filters_tensor.data = new_filters_data

    conv2d_layer:TfliteConv2dLayer = tflite_model.layers[0]
    assert np.array_equal(new_filters_data, conv2d_layer.filters_data)


def test_set_tensor_data_with_save():
    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)

    conv2d_layer:TfliteConv2dLayer = tflite_model.layers[0]
    older_filters_data = conv2d_layer.filters_data
    new_filters_data = np.random.randint(0, 127, size=older_filters_data.shape, dtype=older_filters_data.dtype)

    conv2d_layer.filters_tensor.data = new_filters_data

    tmp_tflite_path = f'{create_tempdir()}/cifar10_resnet_v1.tflite'
    tflite_model.save(tmp_tflite_path)

    try:
        loaded_tflite_model = TfliteModel.load_flatbuffer_file(tmp_tflite_path)
        loaded_conv2d_layer:TfliteConv2dLayer = loaded_tflite_model.layers[0]
        assert np.array_equal(new_filters_data, loaded_conv2d_layer.filters_data)
    finally:
        os.remove(tmp_tflite_path)


def test_conv2d_params():
    from mltk.core.tflite_micro import TfliteMicro

    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)
    recorded_layers = TfliteMicro.record_model(IMAGE_CLASSIFICATION_TFLITE_PATH)

    for layer_index, recorded_data in enumerate(recorded_layers):
        tf_layer = tflite_model.layers[layer_index]

        if not isinstance(tf_layer, TfliteConv2dLayer):
            continue

        expected_params = recorded_data.metadata['params']
        expected_params['per_channel_multiplier'] = np.frombuffer(expected_params['per_channel_multiplier'], dtype=np.int32)
        expected_params['per_channel_shift'] = np.frombuffer(expected_params['per_channel_shift'], dtype=np.int32)

        calc_params = tf_layer.params
    
        assert calc_params.padding.value == expected_params['padding_type']
        assert calc_params.padding.width == expected_params['padding_width']
        assert calc_params.padding.height == expected_params['padding_height']
        assert calc_params.stride_width == expected_params['stride_width']
        assert calc_params.stride_height == expected_params['stride_height']
        assert calc_params.dilation_width_factor == expected_params['dilation_width_factor']
        assert calc_params.dilation_height_factor == expected_params['dilation_height_factor']
        assert calc_params.input_offset == expected_params['input_offset']
        assert calc_params.weights_offset == expected_params['weights_offset']
        assert calc_params.output_offset == expected_params['output_offset']
        assert calc_params.quantized_activation_min == expected_params['quantized_activation_min']
        assert calc_params.quantized_activation_max == expected_params['quantized_activation_max']
        assert np.allclose(calc_params.per_channel_output_multiplier, expected_params['per_channel_multiplier'])
        assert np.allclose(calc_params.per_channel_output_shift, expected_params['per_channel_shift'])


def test_depthwise_conv2d_params():
    from mltk.core.tflite_micro import TfliteMicro
    from mltk.core import load_tflite_model

    tflite_model = load_tflite_model('keyword_spotting')
    recorded_layers = TfliteMicro.record_model(tflite_model.path)

    for layer_index, recorded_data in enumerate(recorded_layers):
        tf_layer = tflite_model.layers[layer_index]

        if not isinstance(tf_layer, TfliteDepthwiseConv2dLayer):
            continue

        expected_params = recorded_data.metadata['params']
        expected_params['per_channel_multiplier'] = np.frombuffer(expected_params['per_channel_multiplier'], dtype=np.int32)
        expected_params['per_channel_shift'] = np.frombuffer(expected_params['per_channel_shift'], dtype=np.int32)

        calc_params = tf_layer.params
    
        assert calc_params.depth_multiplier == expected_params['depth_multiplier']
        assert calc_params.padding.value == expected_params['padding_type']
        assert calc_params.padding.width == expected_params['padding_width']
        assert calc_params.padding.height == expected_params['padding_height']
        assert calc_params.stride_width == expected_params['stride_width']
        assert calc_params.stride_height == expected_params['stride_height']
        assert calc_params.dilation_width_factor == expected_params['dilation_width_factor']
        assert calc_params.dilation_height_factor == expected_params['dilation_height_factor']
        assert calc_params.input_offset == expected_params['input_offset']
        assert calc_params.weights_offset == expected_params['weights_offset']
        assert calc_params.output_offset == expected_params['output_offset']
        assert calc_params.quantized_activation_min == expected_params['quantized_activation_min']
        assert calc_params.quantized_activation_max == expected_params['quantized_activation_max']
        assert np.allclose(calc_params.per_channel_output_multiplier, expected_params['per_channel_multiplier'])
        assert np.allclose(calc_params.per_channel_output_shift, expected_params['per_channel_shift'])


def test_fully_connected_params():
    from mltk.core.tflite_micro import TfliteMicro
    from mltk.core import load_tflite_model

    tflite_model = load_tflite_model('anomaly_detection')
    recorded_layers = TfliteMicro.record_model(tflite_model.path)

    for layer_index, recorded_data in enumerate(recorded_layers):
        tf_layer = tflite_model.layers[layer_index]

        if not isinstance(tf_layer, TfliteFullyConnectedLayer):
            continue

        expected_params = recorded_data.metadata['params']
        calc_params = tf_layer.params
    
        assert calc_params.input_offset == expected_params['input_offset']
        assert calc_params.weights_offset == expected_params['weights_offset']
        assert calc_params.output_offset == expected_params['output_offset']
        assert calc_params.output_multiplier == expected_params['output_multiplier']
        assert calc_params.output_shift == expected_params['output_shift']
        assert calc_params.quantized_activation_min == expected_params['quantized_activation_min']
        assert calc_params.quantized_activation_max == expected_params['quantized_activation_max']


def test_average_pool_params():
    from mltk.core.tflite_micro import TfliteMicro
    from mltk.core import load_tflite_model

    tflite_model = load_tflite_model('keyword_spotting')
    recorded_layers = TfliteMicro.record_model(tflite_model.path)

    for layer_index, recorded_data in enumerate(recorded_layers):
        tf_layer = tflite_model.layers[layer_index]

        if not isinstance(tf_layer, TflitePooling2dLayer):
            continue

        expected_params = recorded_data.metadata['params']
        calc_params = tf_layer.params
    
        assert calc_params.padding.value == expected_params['padding_type']
        assert calc_params.padding.width == expected_params['padding_width']
        assert calc_params.padding.height == expected_params['padding_height']
        assert calc_params.stride_width == expected_params['stride_width']
        assert calc_params.stride_height == expected_params['stride_height']
        assert calc_params.quantized_activation_min == expected_params['quantized_activation_min']
        assert calc_params.quantized_activation_max == expected_params['quantized_activation_max']

def test_max_pool_params():
    from mltk.core.tflite_micro import TfliteMicro
    from mltk.core import load_tflite_model

    tflite_model = load_tflite_model('keyword_spotting_on_off')
    recorded_layers = TfliteMicro.record_model(tflite_model.path)

    for layer_index, recorded_data in enumerate(recorded_layers):
        tf_layer = tflite_model.layers[layer_index]

        if not isinstance(tf_layer, TflitePooling2dLayer):
            continue

        expected_params = recorded_data.metadata['params']
        calc_params = tf_layer.params
    
        assert calc_params.padding.value == expected_params['padding_type']
        assert calc_params.padding.width == expected_params['padding_width']
        assert calc_params.padding.height == expected_params['padding_height']
        assert calc_params.stride_width == expected_params['stride_width']
        assert calc_params.stride_height == expected_params['stride_height']
        assert calc_params.quantized_activation_min == expected_params['quantized_activation_min']
        assert calc_params.quantized_activation_max == expected_params['quantized_activation_max']