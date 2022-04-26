"""
This provides an example of how to use this library

To run this script, issue the command:

python example.py
"""
import os
import tempfile


# Required: Import the TfliteModelParameters class
from mltk.core.tflite_model_parameters import TfliteModelParameters

# This is optional, get the path to the default .tflite model file
# used by the unit tests
from mltk.utils.test_helper.data import IMAGE_CLASSIFICATION_TFLITE_PATH



# Update this to the path of your .tflite model file
# Or use the default model that comes with the package
my_tflite_path = IMAGE_CLASSIFICATION_TFLITE_PATH
tmp_my_tflite_path = f'{tempfile.gettempdir()}/{os.path.basename(my_tflite_path)}'


################################
# Begin example code:



# Instantiate the model parameters object
model_params = TfliteModelParameters()

# Add some model parameters:

# Add a bool
model_params['normalize_samples'] = True
# Add an integer
model_params['sample_rate'] = 8000
# Add a float
model_params['period_ms'] = 0.624
# Add a string
model_params['msg'] = 'This is neat!'
# Add a list of strings
model_params['class_labels'] = ['left', 'right', 'up', 'download', 'unknown']
# Add binary data 
model_params['blob'] = b'\xE2\x82\xAC'


# Print a summary of the parameters
print('Model parameters:')
print(model_params)

# Add the model parameters to the .tflite model and save to a new file
model_params.add_to_tflite_file(my_tflite_path, output=tmp_my_tflite_path)

# Load the model parameters from the saved file
loaded_model_params = TfliteModelParameters.load_from_tflite_file(tmp_my_tflite_path)

# Print a summary of the loaded parameters
print('Loaded model parameters:')
print(loaded_model_params)

print('Class labels:')
for label in loaded_model_params['class_labels']:
    print(label)

if loaded_model_params['normalize_samples']:
    print('Normalizing samples')

if loaded_model_params['sample_rate'] == 8000:
    print('Sample rate is 8k')

if loaded_model_params['period_ms']  < 1.0:
    print('Period is less than 1s')