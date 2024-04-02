import os
import sys 

try:
    import imp 
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))


try:
    # This works around the error: module 'tensorflow' has no attribute 'io'
    # when importing tensorflow_lite_support.metadata.schema_py_generated
    
    import tensorflow_lite_support

    tensorflow_lite_support_dir = os.path.dirname(tensorflow_lite_support.__file__)
    metadata_path = f'{tensorflow_lite_support_dir}/metadata/python/metadata.py'
    if os.path.exists(metadata_path):
        data = ''
        updated = False
        with open(metadata_path, 'r') as f:
            for line in f:
                if 'except ImportError as e:' in line:
                    updated = True
                    line = line.replace('ImportError', 'Exception')
                data += line
        if updated:
            with open(metadata_path, 'w') as f:
                f.write(data)
except:
    pass


try:
    # Linux uses this package
    from tensorflow_lite_support.metadata.schema_py_generated import * # pylint: disable=wildcard-import,unused-wildcard-import
except ModuleNotFoundError:
    # While Windows uses this package
    from tflite_support.schema_py_generated import *  # pylint: disable=wildcard-import,unused-wildcard-import
