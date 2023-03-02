"""Rock, Paper, Scissors v1
****************************************

Contains grayscale images of the hand gestures:

- rock
- paper
- scissors
"""
import logging
from mltk.utils.archive_downloader import download_verify_extract

DOWNLOAD_URL = 'https://github.com/SiliconLabs/mltk_assets/raw/master/datasets/rock_paper_scissors.7z'
"""Public download URL"""
VERIFY_SHA1 = '1CE48F66F7FF999958550147D75ABA8DA185280C'
"""SHA1 hash of archive file"""

INPUT_HEIGHT = 96
"""Sample height"""
INPUT_WIDTH = 96
"""Sample width"""
INPUT_DEPTH = 1
"""Sample depth"""
INPUT_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, INPUT_DEPTH)
"""Sample shape"""
CLASSES = ('rock', 'paper', 'scissor')
"""Dataset class labels"""


def load_data(
    dest_dir:str=None,
    dest_subdir='datasets/rock_paper_scissors/v1',
    logger:logging.Logger=None,
    clean_dest_dir=False
):
    """Load the Rock, Paper, Scissors Dataset v1

    Contains 96x96x1 images of the hand gestures:

    - rock
    - paper
    - scissors
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
