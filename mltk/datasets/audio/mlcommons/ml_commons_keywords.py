"""ML Commons Keywords
****************************

Overview
----------

This dataset is a subset of:

- https://mlcommons.org/en/multilingual-spoken-words/
- https://mlcommons.org/en/policies/

It contains 3 samples of each English word converted to 16kHz, PCM audio.

Abstract
-----------

Multilingual Spoken Words Corpus is a large and growing audio dataset of spoken words in 50 languages for academic research and commercial applications in keyword spotting and spoken term search, licensed under CC-BY 4.0. The dataset contains more than 340,000 keywords, totaling 23.4 million 1-second spoken examples (over 6,000 hours).

The dataset has many use cases, ranging from voice-enabled consumer devices to call center automation. All alignments are included in the dataset. Please see our paper for a detailed analysis of the contents of the data and methods for detecting potential outliers, along with baseline accuracy metrics on keyword spotting models trained from our dataset compared to models trained on a manually-recorded keyword dataset.

Read our full paper `here <https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/fe131d7f5a6b38b23cc967316c13dae2-Paper-round2.pdf>`_


"""

import logging
from mltk.utils.archive_downloader import download_verify_extract

DOWNLOAD_URL1 = 'https://www.dropbox.com/s/j4p9w4h92e8rruo/mlcommons_keywords_subset_part1.7z?dl=1'
"""Public download URL for the first part of the dataset"""
VERIFY_URL1_SHA1 = '6f515d8247e2fee70cd0941420918c8fe57a31e8'
"""SHA1 hash of the DOWNLOAD_URL1 archive file"""

DOWNLOAD_URL2 = 'https://www.dropbox.com/s/zacujsccjgk92b2/mlcommons_keywords_subset_part2.7z?dl=1'
"""Public download URL for the second part of the dataset"""
VERIFY_URL2_SHA1 = '7816f5ffa1deeafa9b5b3faae563f44198031796'
"""SHA1 hash of the DOWNLOAD_URL2 archive file"""

def download(
    dest_dir:str,
    logger:logging.Logger=None,
    clean_dest_dir=False,
) -> str:
    """Download and extract the dataset

    Returns:
        The path to the extracted dataset directory
    """
    download_verify_extract(
        url=DOWNLOAD_URL1,
        dest_dir=dest_dir,
        file_hash=VERIFY_URL1_SHA1,
        show_progress=False,
        remove_root_dir=False,
        clean_dest_dir=clean_dest_dir,
        logger=logger
    )

    sample_dir = download_verify_extract(
        url=DOWNLOAD_URL2,
        dest_dir=dest_dir,
        file_hash=VERIFY_URL2_SHA1,
        show_progress=False,
        remove_root_dir=False,
        clean_dest_dir=False,
        logger=logger
    )

    return sample_dir
