
import numpy as np
from keras_preprocessing.image.utils import (
    img_to_array,
    load_img
)
from mltk.core import load_tflite_or_keras_model, load_mltk_model, TfliteModel, KerasModel
from mltk.utils.path import create_user_dir




def main():
    dump_dir = create_user_dir('fingerprint_authenticator/dumps')

    fp0_path = f'{dump_dir}/../processed-14.jpg'
    fp1_path = f'{dump_dir}/processed-0.jpg'

    #fp0_path = 'C:/Users/reed/.mltk/datasets/silabs_fingerprints/v2/processed/57b85337/and_1/raw_and_1_2.jpg'
    #fp1_path = 'C:/Users/reed/.mltk/datasets/silabs_fingerprints/v2/processed/57b85337/and_1/raw_and_1_3.jpg'

    #fp0_path = 'C:/Users/reed/.mltk/datasets/silabs_fingerprints/v2/processed/57b85337/ka_3/raw_ka_3_7.jpg'
    #fp1_path = 'C:/Users/reed/.mltk/datasets/silabs_fingerprints/v2/processed/57b85337/ka_3/raw_ka_3_8.jpg'

    fp0_img = _load_img(fp0_path)
    fp1_img = _load_img(fp1_path)
    
    # fp0_img = fp0_img[6:192-6, 6:192-6]
    #fp1_img = fp1_img[6:192-6, 6:192-6]

    fp0_img = fp0_img - 128
    fp1_img = fp1_img - 128
    fp0_img = fp0_img.astype(np.int8)
    fp1_img = fp1_img.astype(np.int8)

    mltk_model = load_mltk_model('fingerprint_reader')
    keras_model:KerasModel = load_tflite_or_keras_model(mltk_model, model_type='h5')
    tflite_model:TfliteModel = load_tflite_or_keras_model(mltk_model, model_type='tflite')

    keras_fp0_img = np.expand_dims(fp0_img, axis=0)
    keras_fp1_img = np.expand_dims(fp1_img, axis=0)
    keras_s0 = keras_model.predict(keras_fp0_img)
    keras_s1 = keras_model.predict(keras_fp1_img)

    keras_dis = np.sqrt(np.sum(np.square(keras_s0 - keras_s1)))

    s = f'{keras_dis:.4f}\n'
    s += f'{keras_s0}\n'
    s += f'{keras_s1}'
    print(s)


    tflite_s0 = tflite_model.predict(fp0_img, y_dtype=np.float32)
    tflite_s1 = tflite_model.predict(fp1_img, y_dtype=np.float32)

    tflite_dis = np.sqrt(np.sum(np.square(tflite_s0 - tflite_s1)))

    s = f'{tflite_dis:.4f}\n'
    s += f'{tflite_s0}\n'
    s += f'{tflite_s1}'
    print(s)



def _load_img(path):
    img = load_img(path, color_mode='grayscale')
    x = img_to_array(img, dtype='uint8')
    img.close()
    return x



if __name__ == '__main__':
    main()