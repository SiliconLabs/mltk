"""ESC: Dataset for Environmental Sound Classification
**************************************************************

https://github.com/karoldvl/paper-2015-esc-dataset

Abstract
------------

One of the obstacles in research activities concentrating on environmental sound classification is the scarcity of suitable and publicly available datasets.
This paper tries to address that issue by presenting a new annotated collection of 2 000 short clips comprising 50 classes of various common sound events,
and an abundant unified compilation of 250 000 unlabeled auditory excerpts extracted from recordings available through the Freesound project.
The paper also provides an evaluation of human accuracy in classifying environmental sounds and compares it to the performance of selected baseline classifiers
using features derived from mel-frequency cepstral coefficients and zero-crossing rate.

Citing
---------

K. J. Piczak. **ESC: Dataset for Environmental Sound Classification**. In *Proceedings of the 23rd ACM international conference on Multimedia*, pp. 1015-1018, ACM, 2015.

"""

import os
import logging

from mltk.utils.archive_downloader import download_verify_extract
from mltk.utils.process_pool import ProcessPool
from mltk.core.preprocess.utils import audio as audio_utils


DOWNLOAD_URL = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
"""Public download URL"""
VERIFY_SHA1 = '60764EEF4F073D27A766033A47956E23022A2EBC'
"""SHA1 hash of the downloaded archive"""


def download(
    dest_dir:str=None,
    dest_subdir:str='datasets/esc-50',
    sample_rate_hertz:int=16000,
    logger:logging.Logger=None,
    clean_dest_dir=False
) -> str:
    """Download the dataset, extract, and convert each sample to the specified data rate in-place.

    Returns:
        The path to the extract and re-sample dataset directory
    """
    if dest_dir:
        dest_subdir = None

    base_dir, is_up_to_date = download_verify_extract(
        dest_dir=dest_dir,
        dest_subdir=dest_subdir,
        url=DOWNLOAD_URL,
        file_hash=VERIFY_SHA1,
        archive_fname='ESC-50.zip',
        remove_root_dir=True,
        clean_dest_dir=clean_dest_dir,
        logger=logger,
        return_uptodate=True
    )

    audio_dir = f'{base_dir}/audio'

    if not is_up_to_date and sample_rate_hertz != 44100:
        with ProcessPool(
            entry_point=_convert_sample_rate,
            n_jobs=8
        ) as pool:
            filenames = list(fn for fn in os.listdir(audio_dir) if fn.endswith('.wav'))
            batch = pool.create_batch(len(filenames))
            for fn in filenames:
                p = f'{audio_dir}/{fn}'
                pool.process(path=p, sample_rate_hz=sample_rate_hertz, pool_batch=batch)
            batch.wait()

    return audio_dir


def _convert_sample_rate(path:str, sample_rate_hz:int):
    sample, original_sample_rate = audio_utils.read_audio_file(
        path,
        return_sample_rate=True,
        return_numpy=True
    )
    sample = audio_utils.resample(
        sample,
        orig_sr=original_sample_rate,
        target_sr=sample_rate_hz
    )
    audio_utils.write_audio_file(
        path,
        sample,
        sample_rate=sample_rate_hz
    )

if __name__ == '__main__':
    dst_dir = download(None)
    print(dst_dir)