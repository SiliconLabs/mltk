"""BRD2601 Background Noise
****************************

This is an audio clip of "silence" recorded by the microphone on the `BRD2601 <https://www.silabs.com/development-tools/wireless/efr32xg24-dev-kit>`_
development board. This may be mixed with dataset audio samples to help simulate
what would be captured by the development's microphone.

"""

import logging
from mltk.core.preprocess.utils import audio as audio_utils

from .. import utils


DOWNLOAD_URL = 'https://github.com/SiliconLabs/mltk_assets/raw/master/datasets/brd2601_background_audio.7z'
"""The public download URL"""
VERIFY_SHA1 = '3069A85002965A7830C660343C215EDD4FAE39C6'
"""The SHA1 hash of the downloaded archive"""

def download(
    dest_dir:str,
    sample_rate_hertz:int=16000,
    logger:logging.Logger=None,
    clean_dest_dir=False
) -> str:
    """Download and return the path to the audio sample"""
    sample_dir = utils.download_and_extract(
        dest_dir=dest_dir,
        urls=(DOWNLOAD_URL, VERIFY_SHA1),
        clean_dest_dir=clean_dest_dir,
        logger=logger
    )[0]

    sample_path = f'{sample_dir}/brd2601_background_audio.wav'

    if sample_rate_hertz != 16000:
        sample, original_sample_rate = audio_utils.read_audio_file(
            sample_path,
            return_sample_rate=True,
            return_numpy=True
        )
        sample = audio_utils.resample(
            sample,
            orig_sr=original_sample_rate,
            target_sr=sample_rate_hertz
        )
        audio_utils.write_audio_file(
            sample_path,
            sample,
            sample_rate=sample_rate_hertz
        )

    return sample_path
