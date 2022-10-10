try:
    from tflite_support.schema_py_generated import *
except ModuleNotFoundError:
    import os
    import tensorflow_lite_support

    tensorflow_lite_support_dir = os.path.dirname(tensorflow_lite_support.__file__)
    metadata_path = f'{tensorflow_lite_support_dir}/metadata/python/metadata.py'
    if os.path.exists(metadata_path):
        # This works around the error: module 'tensorflow' has no attribute 'io'
        # when importing tensorflow_lite_support.metadata.schema_py_generated
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

    from tensorflow_lite_support.metadata.schema_py_generated import *