"""Rock, Paper, Scissors v2
****************************************

Contains grayscale images of the hand gestures:

- rock
- paper
- scissors
- _unknown_
"""

from mltk.utils.archive_downloader import download_verify_extract

DOWNLOAD_URL = 'https://github.com/SiliconLabs/mltk_assets/raw/master/datasets/rock_paper_scissors_v2.7z'
VERIFY_SHA1 = '77ED1793BE7871DCAA79D935B39BA4D23A28E2C3'


INPUT_HEIGHT = 96
INPUT_WIDTH = 96
INPUT_DEPTH = 1
INPUT_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, INPUT_DEPTH)
CLASSES = ('rock', 'paper', 'scissor', '_unknown_')



def load_data():
    """Load the Rock, Paper, Scissors Dataset v2

    Contains 96x96x1 images of the hand gestures:

    - rock
    - paper
    - scissors
    - _unknown_
    """

    path = download_verify_extract(
        url=DOWNLOAD_URL,
        dest_subdir='datasets/rock_paper_scissors/v2',
        file_hash=VERIFY_SHA1,
        show_progress=True,
        remove_root_dir=True
    )

    return path
