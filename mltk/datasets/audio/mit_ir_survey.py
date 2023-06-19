"""MIT Impulse Response Survey
=================================

This is a dataset of environmental impulse responses from various real-world locations,
collected in the `MIT IR Survey <https://mcdermottlab.mit.edu/Reverb/IR_Survey.html>`_ by Traer and McDermott.

Each audio file is a waveform which contains the impulse response of a location.
That is to say, how an instantaneous pressure at $t = 0$ is reflected, damped and scattered in the environment.

By convolving the dataset impulse responses with an audio clip,
we can simulate how that audio clip would sound if emitted and recorded in the environment
where the impulse response was recorded. This is a technique commonly used for data augmentation in
audio processing problems, commonly referred to as
multi-style training (see `Deep Spoken Keyword Spotting: An Overview <https://arxiv.org/pdf/2111.10592.pdf>`_), simulated reverberation
(see e.g. `End-to-End Streaming Keyword Spotting <https://arxiv.org/pdf/1812.02802.pdf>`_) or acoustic simulation.

License
---------

CC-BY 4.0 (see `MIT Creative Commons License <https://creativecommons.org/licenses/by/4.0>`_ for details).

Credits
--------------

Traer and McDermott 2016 paper `Statistics of natural reverberation enable perceptual separation of sound and space <https://www.pnas.org/doi/full/10.1073/pnas.1612524113>`_

"""
from typing import List
import os
import logging
import numpy as np

from mltk.core.preprocess.utils import audio as audio_utils
from mltk.utils.python import append_exception_msg
from mltk.utils.archive_downloader import download_verify_extract





DOWNLOAD_URL = 'https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip'
"""Public download URL"""
VERIFY_SHA1 = 'de04f5be419c12f4f847f65d7ef8e2356b73aa38'
"""SHA1 hash of the downloaded archive file"""



def download(
    dest_dir:str=None,
    dest_subdir='datasets/mit_ir_survey',
    logger:logging.Logger=None,
    clean_dest_dir=False,
    sample_rate_hz=16000,
) -> str:
    """Download and extract the dataset

    Returns:
        The directory path to the extracted dataset
    """
    try:
        import soundfile
    except ModuleNotFoundError as e:
        append_exception_msg(e, 'Try running the command: pip install soundfile')
        raise e

    if dest_dir:
        dest_subdir = None

    sample_dir = download_verify_extract(
        url=DOWNLOAD_URL,
        archive_fname='mit_ir_survey.zip',
        dest_dir=dest_dir,
        dest_subdir=dest_subdir,
        file_hash=VERIFY_SHA1,
        show_progress=False,
        remove_root_dir=False,
        clean_dest_dir=clean_dest_dir,
        logger=logger
    )

    src_dir = f'{sample_dir}/Audio'
    for fn in os.listdir(src_dir):
        if not fn.endswith('.wav'):
            continue

        dst_path = f'{sample_dir}/{fn}'
        if not os.path.exists(dst_path):
            data, sr = soundfile.read(f'{src_dir}/{fn}')
            data = data.astype(np.float32)
            if sr != sample_rate_hz:
                data = audio_utils.resample(data, orig_sr=sr, target_sr=sample_rate_hz)
            audio_utils.write_audio_file(dst_path, data, sample_rate=sample_rate_hz)

    return sample_dir



def apply_ir(
    audio: np.ndarray,
    ir: np.ndarray
) -> np.ndarray:
    """Apply an impulse response to the given audio sample"""
    try:
        from scipy import signal
    except ModuleNotFoundError as e:
        append_exception_msg(e, 'Try running the command: pip install scipy')
        raise

    return signal.fftconvolve(audio, ir)


def load_dataset(dataset_dir:str) -> List[np.ndarray]:
    """Load the impulse response dataset directory into RAM"""
    try:
        import soundfile
    except ModuleNotFoundError as e:
        append_exception_msg(e, 'Try running the command: pip install soundfile')
        raise e

    retval = []
    for fn in os.listdir(dataset_dir):
        if not fn.endswith('.wav'):
            continue
        data, _ = soundfile.read(f'{dataset_dir}/{fn}')
        data = data.astype(np.float32)
        retval.append(data)

    return retval



def apply_random_ir(
    audio: np.ndarray,
    ir_samples:List[np.ndarray],
    seed:int=42
) -> np.ndarray:
    """Appyly a random impulse response to the given audio sample"""
    rgen = np.random.RandomState(seed=seed)
    index = rgen.choice(len(ir_samples))
    ir = ir_samples[index]

    return apply_ir(audio, ir)