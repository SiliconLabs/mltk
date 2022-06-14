import os 


_curdir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')

IMAGE_EXAMPLE1_TFLITE_PATH = f'{_curdir}/image_example1.tflite' 
IMAGE_EXAMPLE1_H5_PATH = f'{_curdir}/image_example1.h5' 
TFLITE_MICRO_SPEECH_TFLITE_PATH = f'{_curdir}/tflite_micro_speech.tflite' 
IMAGE_CLASSIFICATION_TFLITE_PATH = f'{_curdir}/image_classification.tflite' 
ANOMALY_DETECTION_TFLITE_PATH = f'{_curdir}/anomaly_detection.tflite' 
TEST_MODEL_WEIGHTS = f'{_curdir}/test_model_weights.h5'