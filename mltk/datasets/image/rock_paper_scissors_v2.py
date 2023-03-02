"""Rock, Paper, Scissors v2
****************************************

Contains grayscale images of the hand gestures:

- rock
- paper
- scissors
- _unknown_
"""

import logging
from mltk.utils.archive_downloader import download_verify_extract

DOWNLOAD_URL = 'https://github.com/SiliconLabs/mltk_assets/raw/master/datasets/rock_paper_scissors_v2.7z'
"""Public download URL"""
VERIFY_SHA1 = '77ED1793BE7871DCAA79D935B39BA4D23A28E2C3'
"""SHA1 hash of archive file"""

INPUT_HEIGHT = 96
"""Sample height"""
INPUT_WIDTH = 96
"""Sample width"""
INPUT_DEPTH = 1
"""Sample depth"""
INPUT_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, INPUT_DEPTH)
"""Sample shape"""
CLASSES = ('rock', 'paper', 'scissor', '_unknown_')
"""Dataset class labels"""



def load_data(
    dest_dir:str=None,
    dest_subdir='datasets/rock_paper_scissors/v2',
    logger:logging.Logger=None,
    clean_dest_dir=False
):
    """Load the Rock, Paper, Scissors Dataset v2

    Contains 96x96x1 images of the hand gestures:

    - rock
    - paper
    - scissors
    - _unknown_
    """
    if dest_dir:
        dest_subdir = None

    path = download_verify_extract(
        url=DOWNLOAD_URL,
        file_hash=VERIFY_SHA1,
        dest_dir=dest_dir,
        dest_subdir=dest_subdir,
        remove_root_dir=True,
        clean_dest_dir=clean_dest_dir,
        logger=logger
    )

    return path
