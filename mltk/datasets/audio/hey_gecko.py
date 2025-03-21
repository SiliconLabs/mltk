"""Hey Gecko
=======================

This is a `synthetically <https://siliconlabs.github.io/mltk/mltk/tutorials/synthetic_audio_dataset_generation.html>`_ generated dataset with the keyword:

- **Hey Gecko**
- **Unknown**

The samples are 16kHz, 16-bit PCM ``.wav`` files.


.. seealso::

    - `AudioDatasetGenerator <https://siliconlabs.github.io/mltk/docs/python_api/utils/audio_dataset_generator/index.html>`_
    - `Synthetic Audio Dataset Generation Tutorial <https://siliconlabs.github.io/mltk/mltk/tutorials/synthetic_audio_dataset_generation.html>`_

"""

import logging
import os
import json

from mltk.utils.archive_downloader import download_verify_extract
from mltk.utils.path import create_user_dir, fullpath
from mltk.utils.audio_dataset_generator import (
    AudioDatasetGenerator,
    Keyword,
    Augmentation,
    VoiceRate,
    VoicePitch,
)




DOWNLOAD_URL = 'https://www.silabs.com/public/files/github/mltk/datasets/sl_synthetic_hey_gecko2.7z'
"""Public download URL"""
VERIFY_SHA1 = '69064ca13f7c4c1e20b7672162e204851c64aef6'
"""SHA1 hash of the downloaded archive file"""

CLASSES = [
    'heygecko',
    '_unknown_',
]
"""The class labels of the dataset samples"""

def download(
    dest_dir:str=None,
    dest_subdir='datasets/hey_gecko',
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




def generate_dataset(out_dir:str=None):
    """Generate the dataset

    This generates the dataset using the `AudioDatasetGenerator <https://siliconlabs.github.io/mltk/docs/python_api/utils/audio_dataset_generator/index.html>`_
    Python package provided by the MLTK.

    """
    import tqdm

    KEYWORDS = [
        Keyword('hey gecko', max_count=12000),
        Keyword('_unknown_', max_count=12000,
        aliases=(
            'ah', 'aah', 'a', 'o', 'uh', 'ee', 'aww',
            'echo', 'leto', 'hey', 'gay', 'yay', 'nay',
            'may', 'grey', 'pay', 'petco', 'hecko', 'get',
            'ghetto',
        ))
    ]

    AUGMENTATIONS = [
        Augmentation(rate=VoiceRate.medium_slow, pitch=VoicePitch.low),
        Augmentation(rate=VoiceRate.medium_slow, pitch=VoicePitch.medium),
        Augmentation(rate=VoiceRate.medium_slow, pitch=VoicePitch.high),
        Augmentation(rate=VoiceRate.medium, pitch=VoicePitch.low),
        Augmentation(rate=VoiceRate.medium, pitch=VoicePitch.medium),
        Augmentation(rate=VoiceRate.medium, pitch=VoicePitch.high),
        Augmentation(rate=VoiceRate.fast, pitch=VoicePitch.low),
        Augmentation(rate=VoiceRate.fast, pitch=VoicePitch.medium),
        Augmentation(rate=VoiceRate.fast, pitch=VoicePitch.high),
    ]

    out_dir = out_dir or create_user_dir('datasets/generated/hey_gecko2')

    with AudioDatasetGenerator(
        out_dir=out_dir,
        n_jobs=8
    ) as generator:
        # Load the cloud backends, installing the Python packages if necessary

        # See: https://codelabs.developers.google.com/codelabs/cloud-text-speech-python3
        if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
            try:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = fullpath(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
            except:
                pass
            with open(os.environ['GOOGLE_APPLICATION_CREDENTIALS'], 'r') as f:
                credentials = json.load(f)
            os.environ['PROJECT_ID'] = credentials['project_id']
            generator.load_backend('gcp', install_python_package=True)
            print('Loaded GCP backend')
        else:
            print('GOOGLE_APPLICATION_CREDENTIALS env not found, *not* loading GCP backend')

        # See: https://docs.aws.amazon.com/polly/latest/dg/get-started-what-next.html
        if 'AWS_ACCESS_KEY_ID' in os.environ or os.path.exists(os.path.expanduser('~/.aws')):
            generator.load_backend('aws', install_python_package=True)
            print('Loaded AWS backend')
        else:
            print('AWS_ACCESS_KEY_ID env not found, *not* loading AWS backend')

        # See: https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/get-started-text-to-speech?pivots=programming-language-python
        if 'SPEECH_KEY' in os.environ:
            generator.load_backend('azure', install_python_package=True)
            print('Loaded Azure backend')
        else:
            print('SPEECH_KEY env not found, *not* loading Azure backend')

        print('Listing voices ...')
        voices = generator.list_voices()

        # Generate a list of all possible configurations, randomly shuffle, then truncate
        # based on the "max_count" specified for each keyword
        print('Listing configurations ...')
        all_configurations = generator.list_configurations(
            keywords=KEYWORDS,
            augmentations=AUGMENTATIONS,
            voices=voices,
            truncate=True,
            seed=42
        )

        n_configs = sum(len(x) for x in all_configurations.values())

        # Print a summary of the configurations
        print(generator.get_summary(all_configurations))

        input(
            '\nWARNING: Running this script is NOT FREE!\n\n'
            'Each cloud backend charges a different rate per character.\n'
            'The character counts are listed above.\n\n'
            'Refer to each backend\'s docs for the latest pricing:\n'
            '- AWS: https://aws.amazon.com/polly/pricing\n'
            '- Azure: https://azure.microsoft.com/en-us/pricing/details/cognitive-services/speech-services\n'
            '- Google: https://cloud.google.com/text-to-speech/pricing\n'
            '\nPress "enter" to continue and generate the dataset\n'
        )

        # Generate the dataset (with pretty progress bars)
        print(f'Generating keywords at: {generator.out_dir}\n')
        with tqdm.tqdm(total=n_configs, desc='Overall'.rjust(10), unit='word', position=1) as pb_outer:
            for keyword, config_list in all_configurations.items():
                with tqdm.tqdm(desc=keyword.value.rjust(10), total=len(config_list), unit='word', position=0) as pb_inner:
                    for config in config_list:
                        generator.generate(
                            config,
                            on_finished=lambda _: (pb_inner.update(1), pb_outer.update(1))
                        )
                    generator.join() # Wait for the current keyword to finish before continuing to the next


if __name__ == '__main__':
    generate_dataset()
