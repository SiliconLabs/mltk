import struct
import sys
import numpy as np


from mltk.utils.jlink_stream import JlinkStream
from mltk.utils.python import install_pip_package
from mltk.core import load_mltk_model, load_tflite_or_keras_model, TfliteModel, TfliteModelParameters


install_pip_package('opencv-python', 'cv2')

from cv2 import cv2 



def main(model:str):
    rescaler = None
    normalize_mean_and_std = False
    tflite_model = None

    if model is not None:
        if not model.endswith(('.tflite', '.mltk.zip')):
            mltk_model = load_mltk_model(model)
            tflite_model:TfliteModel = load_tflite_or_keras_model(mltk_model, model_type='tflite')
        else:
            tflite_model:TfliteModel = load_tflite_or_keras_model(model, model_type='tflite')

        tflite_model_params = TfliteModelParameters.load_from_tflite_model(tflite_model)

        if 'samplewise_norm.rescale' in tflite_model_params:
            rescaler = tflite_model_params['samplewise_norm.rescale']
        if 'samplewise_norm.mean_and_std' in tflite_model_params:
            normalize_mean_and_std = tflite_model_params['samplewise_norm.mean_and_std']


    with JlinkStream() as jlink:
        image_stream = jlink.open('image')

        while True:
            header_bytes = image_stream.read_all(10, timeout=-1)
            img_length, width, height, channels, dtype = struct.unpack('<LHHBB', header_bytes)
            if img_length == 0 or img_length > 256*256*3:
                continue

            img_bytes = image_stream.read_all(img_length, timeout=-1)
            img_buffer = np.frombuffer(img_bytes, dtype=np.uint8)

            if tflite_model is not None:
                test_img = np.reshape(img_buffer, (height, width, channels))
                if rescaler:
                    test_img = test_img * rescaler
                    test_img = test_img.astype(np.float32)
                elif normalize_mean_and_std:
                    test_img = (test_img - np.mean(test_img, keepdims=True)) / np.std(test_img, keepdims=True)
                    test_img = test_img.astype(np.float32)

                preds = tflite_model.predict(test_img)
                print(f'Predictions: {preds}')


            if channels == 1:
                img = np.reshape(img_buffer, (height, width))
            else:
                img = np.reshape(img_buffer, (height, width, channels))


            cv2.imshow('image', img)
            cv2.waitKey(1)


if __name__ == '__main__':
    model = None if len(sys.argv) == 1 else sys.argv[1]
    main(model)