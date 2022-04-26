"""
This provides an example of how to use this library

To run this script, issue the command:

python example.py
"""
import tempfile


# Required: Import the TfliteModel class
from mltk.core import TfliteModel

# This is optional, get the path to the default .tflite model file
# used by the unit tests
from mltk.utils.test_helper.data import IMAGE_CLASSIFICATION_TFLITE_PATH



# Update this to the path of your .tflite model file
# Or use the default model that comes with the package
my_tflite_path = IMAGE_CLASSIFICATION_TFLITE_PATH


# Load the .tflite
tflite_model = TfliteModel.load_flatbuffer_file(my_tflite_path)

# Print a summary of the model
print('Model summary:')
print(tflite_model.summary())


# Iterate through each layer of the model
print('Model layers')
for layer in tflite_model.layers:
    # See TfliteLayer for additional info
    print(layer)
    print('  Inputs')
    for i in layer.inputs:
        print(f'   {i}')
    print('  Outputs')
    for o in layer.outputs:
        print(f'   {o}')

# Update the model's description
# This updates the .tflite's "description" field (which will be displayed in GUIs like https://netron.app)
tflite_model.description = "My awesome model"
print(f'New model description: {tflite_model.description}')

# Save a new .tflite with the updated description
new_model_path = f'{tempfile.gettempdir()}/my_new_model.tflite'
tflite_model.save(new_model_path)


# Add some metadata to the .tflite
metadata = 'this is metadata'.encode('utf-8')
tflite_model.add_metadata('my_metadata', metadata)

# Retrieve all the metadata in the .tflite
all_metadata = tflite_model.get_all_metadata()
print('Model metadata:')
for key, data in all_metadata.items():
    print(f'{key}: length={len(data)} bytes')

# Save a new .tflite with the updated metadata
new_model_path = f'{tempfile.gettempdir()}/my_new_model.tflite'
tflite_model.save(new_model_path)



# You must have Tensorflow instance to perform this step
# This will run inference with the given buffer and return 
# the results. The input_buffer can be:
# - a single sample as a numpy array
# - a numpy array of 1 or more samples
# - A Python generator that returns (batch_x, batch_y)
# inference_results = tflite_model.predict(..)