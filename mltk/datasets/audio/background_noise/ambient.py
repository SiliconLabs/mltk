"""Generic Background Noise
****************************

This provides a list of generic background noise samples from https://mixkit.co/free-sound-effects/public-places

License
----------

https://mixkit.co/license/#sfxFree



"""
from typing import List, Union
import logging

from .. import utils


DOWNLOAD_URLS = [
    'https://assets.mixkit.co/sfx/download/mixkit-very-crowded-pub-or-party-loop-360.wav',
    'https://assets.mixkit.co/sfx/download/mixkit-big-crowd-talking-loop-364.wav',
    'https://assets.mixkit.co/sfx/download/mixkit-restaurant-crowd-talking-ambience-444.wav',
    'https://assets.mixkit.co/sfx/download/mixkit-keyboard-typing-1386.wav',
    'https://assets.mixkit.co/sfx/download/mixkit-office-ambience-447.wav',
    'https://assets.mixkit.co/sfx/download/mixkit-hotel-lobby-with-dining-area-ambience-453.wav'
]
"""The background noise sample download URLs"""


def download(
    dest_dir:str,
    sample_rate_hertz:int=16000,
    urls:Union[str,List[str]]=None,
    logger:logging.Logger=None
) -> List[str]:
    """Download the sample and return a list of file paths"""
    # See https://mixkit.co

    return utils.download(
        dest_dir=dest_dir,
        urls=DOWNLOAD_URLS,
        sample_rate_hertz=sample_rate_hertz
    )
