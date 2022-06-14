import os
import shutil
import pytest
from mltk.core import TfliteModel, TfliteModelParameters
from mltk.utils.test_helper.data import IMAGE_CLASSIFICATION_TFLITE_PATH
from mltk.utils.path import create_tempdir



tmp_tflite_path = f'{create_tempdir()}/{os.path.basename(IMAGE_CLASSIFICATION_TFLITE_PATH)}'





def test_add_to_tflite_file_api():
    shutil.copy(IMAGE_CLASSIFICATION_TFLITE_PATH, tmp_tflite_path)

    params = TfliteModelParameters(dict(
        test=1,
        names=['a', 'b', 'c'],
        foo='bar'
    ))

    params.add_to_tflite_file(tmp_tflite_path)

    new_params = TfliteModelParameters.load_from_tflite_file(tmp_tflite_path)

    assert params == new_params

def test_add_to_tflite_file2_api():
    shutil.copy(IMAGE_CLASSIFICATION_TFLITE_PATH, tmp_tflite_path)

    params = TfliteModelParameters(dict(
        test=1,
        names=['a', 'b', 'c'],
        foo='bar'
    ))

    tmp_tflite_path2 = tmp_tflite_path + '2'
    _remove_file(tmp_tflite_path2)
    params.add_to_tflite_file(tmp_tflite_path, output=tmp_tflite_path2)

    new_params = TfliteModelParameters.load_from_tflite_file(tmp_tflite_path2)

    assert params == new_params


def test_add_to_tflite_flatbuffer_api():
    params = TfliteModelParameters(dict(
        test=1,
        names=['a', 'b', 'c'],
        foo='bar'
    ))

    with open(IMAGE_CLASSIFICATION_TFLITE_PATH, 'rb') as fp:
        tflite_flatbuffer = fp.read()

    updated_flatbuffer = params.add_to_tflite_flatbuffer(tflite_flatbuffer)

    new_params = TfliteModelParameters.load_from_tflite_flatbuffer(updated_flatbuffer)

    assert params == new_params


def test_add_to_tflite_model_api():
    params = TfliteModelParameters(dict(
        test=1,
        names=['a', 'b', 'c'],
        foo='bar'
    ))

    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)

    params.add_to_tflite_model(tflite_model)

    new_params = TfliteModelParameters.load_from_tflite_model(tflite_model)

    assert params == new_params


def test_setitem_api_with_good_data():
    params = TfliteModelParameters()

    params['a'] = True 
    params['b'] = 1 
    params['c'] = -1 
    params['d'] = 3.13
    params['e'] = 'test'
    params['f'] = ['1', '2', '3']
    params['g'] = ('1', '2', '3')
    params['h'] = b'\x11\x22\x33'
    params['i'] = [1, 2, 3, 4]
    params['j'] = [1., 2., 3., 4.]

    serialized_params = params.serialize()

    deserialized_params = TfliteModelParameters.deserialize(serialized_params)

    for key, value in deserialized_params.items():
        if isinstance(value, float):
            assert abs(value - params[key]) < 1e-3
        elif isinstance(params[key], tuple):
            assert value == list(params[key])
        else:
            assert value == params[key]

def test_setitem_api_with_bad_data():
    params1 = TfliteModelParameters()
    with pytest.raises(ValueError):
        params1['a'] = {'bad': 1}

    tflite_model = TfliteModel.load_flatbuffer_file(IMAGE_CLASSIFICATION_TFLITE_PATH)
    params2 = TfliteModelParameters()
    with pytest.raises(ValueError):
        params2['a'] = tflite_model

    params3 = TfliteModelParameters()
    with pytest.raises(ValueError):
        params3['a'] = ['a', 2, 3]

    params4 = TfliteModelParameters()
    with pytest.raises(ValueError):
        params4['a'] = 2**64

    params5 = TfliteModelParameters()
    with pytest.raises(ValueError):
        params5['a'] = [2, 3.]


def test_put_api_with_good_data():
    params = TfliteModelParameters()

    params.put('a', True, dtype='bool') 
    params.put('b', 567, dtype='uint32')
    params.put('c', -234, dtype='int16')
    params.put('d', 3.14, dtype='float')
    params.put('e', 'test msg', dtype='str')
    params.put('f',  ['1', '2', '3'], dtype='str_list')
    params.put('g', ('1', '2', '3'), dtype='str_list')
    params.put('h', b'\x11\x22\x33', dtype='bin')
    params.put('i', (1, 2, 3), dtype='int32_list')
    params.put('j', (1., 2., 3.), dtype='float_list')

    serialized_params = params.serialize()

    deserialized_params = TfliteModelParameters.deserialize(serialized_params)

    for key, value in deserialized_params.items():
        if isinstance(value, float):
            assert abs(value - params[key].value) < 1e-3
        elif isinstance(params[key].value, tuple):
            assert value == list(params[key].value)
        else:
            assert value == params[key].value

def test_put_api_with_bad_data():

    params = TfliteModelParameters()
    with pytest.raises(ValueError):
        params.put('a', 1, dtype='foo')
    with pytest.raises(ValueError):
        params.put('a', [22], dtype='int32')
    with pytest.raises(ValueError):
        params.put('a', 'strr', dtype='uint8')
    with pytest.raises(ValueError):
        params.put('a', 'asdf', dtype='float')
    with pytest.raises(ValueError):
        params.put('a', 'asdf', dtype='double')
    with pytest.raises(ValueError):
        params.put('a', dict(d=2), dtype='str_list')
    with pytest.raises(ValueError):
        params.put('a', dict(d=2), dtype='bin')
    with pytest.raises(ValueError):
        params.put('a', dict(d=2.), dtype='int32_list')
    with pytest.raises(ValueError):
        params.put('a', dict(d=2), dtype='float_list')


def _remove_file(p):
    if os.path.exists(p):
        os.remove(p)
