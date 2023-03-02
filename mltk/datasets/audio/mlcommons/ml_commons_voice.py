"""ML Commons Voice Subset
****************************

Overview
----------

This dataset is a subset of:

- https://commonvoice.mozilla.org/en/datasets
- https://commonvoice.mozilla.org/en/terms
- https://creativecommons.org/publicdomain/zero/1.0/


This contains short clips of different people reading.
The clips have been converted to 16kHz, PCM audio.

Abstract
-----------

Common Voice is a publicly available voice dataset, powered by the voices of volunteer contributors around the world. People who want to build voice applications can use the dataset to train machine learning models.


At present, most voice datasets are owned by companies, which stifles innovation. Voice datasets also underrepresent: non-English speakers, people of colour, disabled people, women and LGBTQIA+ people. This means that voice-enabled technology doesn’t work at all for many languages, and where it does work, it may not perform equally well for everyone. We want to change that by mobilising people everywhere to share their voice.


"""

import logging
from mltk.utils.archive_downloader import download_verify_extract


DOWNLOAD_URL = 'https://www.dropbox.com/s/l9uxyr22w3jgenc/common_voice_subset.7z?dl=1'
"""Public download URL"""
VERIFY_SHA1 = 'ce424afd5d9b754f3ea6b3a4f78304f48e865f93'
"""SHA1 hash of the dataset archive file"""

def download(
    dest_dir:str,
    logger:logging.Logger=None,
    clean_dest_dir=False
) -> str:
    """Download and extract the dataset

    Returns:
        The path to the extracted dataset directory
    """
    sample_dir = download_verify_extract(
        url=DOWNLOAD_URL,
        dest_dir=dest_dir,
        file_hash=VERIFY_SHA1,
        show_progress=False,
        remove_root_dir=False,
        clean_dest_dir=clean_dest_dir,
        logger=logger
    )

    return sample_dir
