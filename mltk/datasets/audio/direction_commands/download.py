

import logging
from mltk.utils.archive_downloader import download_verify_extract


DOWNLOAD_URL = 'https://www.dropbox.com/s/ulumv0sxbtcssvr/synthetic_direction_commands.7z?dl=1'
"""Public download URL"""
VERIFY_SHA1 = '9e9295b4eda3d9da9cd242063c1646b15908da55'
"""SHA1 hash of the downloaded archive file"""

CLASSES = [
    'left',
    'right',
    'up',
    'down',
    'stop',
    'go'
]
"""The class labels of the dataset samples"""

def download(
    dest_dir:str=None,
    dest_subdir='datasets/direction_commands',
    logger:logging.Logger=None,
    clean_dest_dir=False
) -> str:
    """Download and extract the dataset

    Returns:
        The directory path to the extracted dataset
    """

    if dest_dir:
        dest_subdir = None

    sample_dir = download_verify_extract(
        url=DOWNLOAD_URL,
        dest_dir=dest_dir,
        dest_subdir=dest_subdir,
        file_hash=VERIFY_SHA1,
        show_progress=False,
        remove_root_dir=False,
        clean_dest_dir=clean_dest_dir,
        logger=logger
    )

    return sample_dir
