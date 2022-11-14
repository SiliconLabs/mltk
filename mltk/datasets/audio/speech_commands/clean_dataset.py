import os 
from typing import List
import shutil
import json


import tqdm
import tensorflow as tf
import numpy as np


from mltk.datasets.audio.speech_commands import speech_commands_v2
from mltk.core.preprocess.audio.audio_feature_generator import AudioFeatureGeneratorSettings
from mltk.core.preprocess.utils import tf_dataset
import mltk.core.preprocess.utils.audio as audio_utils
from mltk.core import load_tflite_model
from mltk.core import TfliteModelParameters
from mltk.utils.python import install_pip_package


install_pip_package('noisereduce')
import noisereduce


def clean_dataset(
    model:str,
    classes:List[str],
    dst_dir:str=None
):
    tflite_model = load_tflite_model(model)
    tflite_parameters = TfliteModelParameters.load_from_tflite_model(tflite_model)
    frontend_settings = AudioFeatureGeneratorSettings(**tflite_parameters)
    tflite_model_classes = tflite_parameters['classes']

    src_dataset_dir = speech_commands_v2.load_data()
    dst_dataset_dir = dst_dir or f'{src_dataset_dir}/_cleaned'

    class_counts = {}
    features_ds, labels_ds = tf_dataset.load_audio_directory(
        directory=src_dataset_dir,
        classes=classes,
        return_audio_data=False,
        class_counts=class_counts, 
        list_valid_filenames_in_directory_function=speech_commands_v2.list_valid_filenames_in_directory
    )

    ds = tf.data.Dataset.zip((features_ds, labels_ds))
    for src_path, class_id in tqdm.tqdm(ds.as_numpy_iterator(), unit='sample', total=sum(class_counts.values())):
        src_path = src_path.decode('utf-8')
        fn = os.path.basename(src_path)
        class_label = classes[class_id]
        tflite_model_class_id = tflite_model_classes.index(class_label)
        sure_dst_dir = f'{dst_dataset_dir}/{class_label}'
        unsure_dst_dir = f'{dst_dataset_dir}/_unsure/{class_label}'
        invalid_dst_dir = f'{dst_dataset_dir}/_invalid/{class_label}'

        os.makedirs(sure_dst_dir, exist_ok=True)
        os.makedirs(unsure_dst_dir, exist_ok=True)
        os.makedirs(invalid_dst_dir, exist_ok=True)

        sample, sr = audio_utils.read_audio_file(src_path, return_sample_rate=True)

        sample = noisereduce.reduce_noise(
            y=sample, 
            sr=sr,
            stationary=True
        )

        if sr != frontend_settings.sample_rate_hz:
            sample = audio_utils.resample(sample, orig_sr=sr, target_sr=frontend_settings.sample_rate_hz)

        # Adjust the audio clip to the length defined in the frontend_settings
        out_length = int((frontend_settings.sample_rate_hz * frontend_settings.sample_length_ms) / 1000)
        adjusted_sample = audio_utils.adjust_length(
            sample,
            out_length=out_length,
            trim_threshold_db=40,
            offset=0
        )

        spectrogram = audio_utils.apply_frontend(
                sample=adjusted_sample, 
                settings=frontend_settings, 
                dtype=tflite_model.outputs[0].dtype
        )
        # The output spectrogram is 2D, add a channel dimension to make it 3D:
        # (height, width, channels=1)
        spectrogram = np.expand_dims(spectrogram, axis=-1)
        spectrogram = np.expand_dims(spectrogram, axis=0)

        preds = tflite_model.predict(spectrogram, verbose=False, y_dtype=np.float32)[0]
        pred_class_id = np.argmax(preds)
        pred = preds[pred_class_id]

        if pred_class_id == tflite_model_class_id:
            if pred > .95:
                shutil.copy(src_path, f'{sure_dst_dir}/{fn}')
            else:
                shutil.copy(src_path, f'{unsure_dst_dir}/{fn}')
        else:
            shutil.copy(src_path, f'{invalid_dst_dir}/{fn}')


def list_valid_samples():
    classes = ['on', 'off', 'left', 'right', 'up', 'down', 'stop', 'go']
    original_dataset_dir = speech_commands_v2.load_data()
    cleaned_dataset_dir = speech_commands_v2.load_clean_data()

    invalid_samples = {}
    for class_label in classes:
        valid_samples = list(os.listdir(f'{cleaned_dataset_dir}/{class_label}'))
        invalid_samples[class_label] = []
        for fn in os.listdir(f'{original_dataset_dir}/{class_label}'):
            if fn not in valid_samples:
                invalid_samples[class_label].append(fn)

    valid_path = os.path.dirname(speech_commands_v2.__file__) + '/invalid_samples.py'
    with open(valid_path, 'w') as f:
        f.write('# This file was auto-generated\n\n')
        f.write('# This contains invalid samples for the following classes:\n')
        for class_label in classes:
            f.write(f'# {class_label}\n')

        f.write('\nINVALID_SAMPLES = ')
        json.dump(invalid_samples, f, indent=2)


if __name__ == '__main__':
    # clean_dataset(
    #     'keyword_spotting_on_off_v2',
    #     classes=('on', 'off')
    # )
    list_valid_samples()