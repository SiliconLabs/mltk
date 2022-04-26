"""Rock, Paper, Scissors v1
****************************************

Contains grayscale images of the hand gestures:

- rock
- paper
- scissors
"""

from mltk.utils.archive_downloader import download_verify_extract

DOWNLOAD_URL = 'https://github.com/SiliconLabs/mltk_assets/raw/master/datasets/rock_paper_scissors.7z'
VERIFY_SHA1 = '1CE48F66F7FF999958550147D75ABA8DA185280C'


INPUT_HEIGHT = 96
INPUT_WIDTH = 96
INPUT_DEPTH = 1
INPUT_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, INPUT_DEPTH)
CLASSES = ('rock', 'paper', 'scissor')



def load_data():
    """Load the Rock, Paper, Scissors Dataset v1

    Contains 96x96x1 images of the hand gestures:

    - rock
    - paper
    - scissors
    """

    path = download_verify_extract(
        url=DOWNLOAD_URL,
        dest_subdir='datasets/rock_paper_scissors/v1',
        file_hash=VERIFY_SHA1,
        show_progress=True,
        remove_root_dir=True
    )
    return path
