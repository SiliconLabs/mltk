"""Utilities for processing audio data"""

import os
from typing import Union
import numpy as np
import tensorflow as tf

from mltk.utils.python import append_exception_msg
from mltk.core.preprocess.audio.audio_feature_generator import (
    AudioFeatureGeneratorSettings,
    AudioFeatureGenerator
)

try:
    import librosa
except Exception as e:
    if os.name != 'nt' and 'sndfile library not found' in f'{e}':
        append_exception_msg(e, '\n\nTry running: sudo apt-get install libsndfile1\n')
    raise


resample = librosa.resample


def read_audio_file(
    path:Union[str,np.ndarray,tf.Tensor],
    return_sample_rate=False,
    return_numpy=True,
    **kwargs
) -> Union[np.ndarray,tf.Tensor]:
    """Reads and decodes an audio file.

    .. note::
        Only mono data is returned as a 1D array/tensor

    Args:
        path: Path to audio file as a python string, numpy string, or tensorflow string
        return_sample_rate: If true then a tuple is returned:  (audio data, audio sample rate)
        return_numpy: If true then return numpy array, else return TF tensor

    Returns:
        If return_sample_rate = False, Audio data as numpy array or TF tensor
        If return_sample_rate = True, (audio data, sample rate)
    """
    raw = tf.io.read_file(path)
    sample, original_sample_rate = tf.audio.decode_wav(
        raw,
        desired_channels=1,

    )
    sample = tf.squeeze(sample, axis=-1)

    if return_numpy:
        sample = sample.numpy()

    if return_sample_rate:
        if return_numpy:
            original_sample_rate = int(original_sample_rate.numpy())

        return sample, original_sample_rate

    return sample


def write_audio_file(
    path:str,
    sample:Union[np.ndarray,tf.Tensor],
    sample_rate:int
) -> Union[str,tf.Tensor]:
    """Write audio data to a file

    Args:
        path: File path to save audio
            If this is does NOT end with .wav, then the path is assumed to be a directory.
            In this case, the audio path is generated as: <path>/<timestamp>.wav
        sample: Audio data to write, if the data type is:
            - ``int16`` then it is converted to float32 and scaled by 32768
        sample_rate: Sample rate of audio
    Returns:
        Path to written file. If this is executing in a non-eager TF function
        then the path is a TF Tensor, otherwise it is a Python string

    """
    if isinstance(sample, np.ndarray):
        sample = tf.convert_to_tensor(sample)

    if len(sample.shape) == 1:
        sample = tf.expand_dims(sample, axis=-1)

    if sample.dtype == np.int16:
        sample = tf.cast(sample, tf.float32)
        sample = sample / 32768.

    wav = tf.audio.encode_wav(sample, sample_rate)

    path = tf.strings.join((os.path.abspath(path),))
    if not tf.strings.regex_full_match(path, r'.*\.wav'):
        ts = tf.timestamp() * 1000
        fn = tf.strings.format('{}.wav', ts)
        path = tf.strings.join((path, fn), separator=os.path.sep)

    tf.io.write_file(path, wav)

    if tf.executing_eagerly() and isinstance(path, tf.Tensor):
        path = path.numpy().decode('utf-8').replace('\\', '/')

    return path


def adjust_length(
    sample:np.ndarray,
    target_sr:int=None,
    original_sr:int=None,
    out_length:int=None,
    offset=0.0,
    trim_threshold_db=30.0,
) -> np.ndarray:
    """Adjust the audio sample length to fit the out_length parameter
    This will audio re-sample the audio to the target sample rate and
    pad with zeros or crop the input sample as necessary.

    Args:
        sample: Audio sample as a numpy array
        target_sr: The sample rate to re-sample the audio. The original_sr arg must also be provided
        original_sr: The original sample rate of teh given audio
        out_length: The length of the output audio sample. If omitted then return the input sample length
        offset: If in_length > out_length, then this is the percentage offset from the beginning of the input to use for the output
            If in_length < out_length, then this is the percentage to pad with zeros before the input sample
        trim_threshold_db: The threshold (in decibels) below reference to consider as silence
    Returns:
        The adjusted audio sample
    """

    if original_sr and original_sr != target_sr:
        sample = librosa.core.resample(sample, orig_sr=original_sr, target_sr=target_sr)

    if trim_threshold_db:
        sample_trimmed, _ = librosa.effects.trim(sample, top_db=int(trim_threshold_db))
    else:
        sample_trimmed = sample

    in_length = sample_trimmed.shape[0]
    if out_length is None:
        out_length = sample.shape[0]

    if len(sample.shape) == 1:
        if in_length > out_length:
            diff = in_length - out_length
            before = int(diff*offset)
            sample = sample_trimmed[before : before + out_length]

        elif in_length < out_length:
            diff = out_length - in_length
            before = int(diff * offset)

            pad_before = np.zeros((before, ), dtype=sample.dtype)
            pad_after = np.zeros((diff - before, ), dtype=sample.dtype)

            sample = np.concatenate((pad_before, sample_trimmed, pad_after), axis=0)

        if sample.shape[0] != out_length:
            sample = sample[:out_length]

    else:
        n_channels = sample_trimmed.shape[1]
        if in_length > out_length:
            diff = in_length - out_length
            before = int(diff*offset)
            sample = sample_trimmed[before : before + out_length, :]

        elif in_length < out_length:
            diff = out_length - in_length
            before = int(diff * offset)

            pad_before = np.zeros((before, n_channels), dtype=sample.dtype)
            pad_after = np.zeros((diff - before, n_channels), dtype=sample.dtype)

            sample = np.concatenate((pad_before, sample_trimmed, pad_after), axis=0)

        if sample.shape[0] != out_length:
            sample = sample[:out_length, :]

    return sample


def apply_frontend(
    sample:np.ndarray,
    settings:AudioFeatureGeneratorSettings,
    dtype=np.float32
) -> np.ndarray:
    """Send the audio sample through the AudioFeatureGenerator and return the generated spectrogram

    Args:
        sample: The audio sample to process in the AudioFeatureGenerator
        settings: The settings to use in the AudioFeatureGenerator
        dtype: The expected audio output data type

    Returns:
        Generated spectrogram of audio
    """

    if np.issubdtype(sample.dtype, np.floating):
        # Convert the floating point data to int16
        # which is what the AudioFeatureGenerator expects it to be
        # sample = librosa.util.normalize(sample, norm=np.inf, axis=None)
        sample = sample * 32768
        sample = sample.astype(np.int16)

    if len(sample.shape) == 2:
        sample = np.squeeze(sample, axis=-1)

    frontend = AudioFeatureGenerator(settings)
    return frontend.process_sample(sample, dtype=dtype)
