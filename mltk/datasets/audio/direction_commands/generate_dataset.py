"""
This script generates a synthetic dataset for the keywords:
- left
- right
- up
- down
- stop
- go

See the corresponding tutorial for more details:
https://siliconlabs.github.io/mltk/mltk/tutorials/synthetic_audio_dataset_generation.html

"""

# Import the necessary Python packages
import os
import json
import tqdm
from mltk.utils.path import create_user_dir, fullpath
from mltk.utils.audio_dataset_generator import (
    AudioDatasetGenerator,
    Keyword,
    Augmentation,
    VoiceRate,
    VoicePitch
)


KEYWORDS = [
    Keyword('left', max_count=10000),
    Keyword('right', max_count=10000),
    Keyword('up', max_count=10000),
    Keyword('down', max_count=10000),
    Keyword('stop', max_count=10000),
    Keyword('go', max_count=10000),
]

AUGMENTATIONS = [
    Augmentation(rate=VoiceRate.xslow, pitch=VoicePitch.low),
    Augmentation(rate=VoiceRate.xslow, pitch=VoicePitch.medium),
    Augmentation(rate=VoiceRate.xslow, pitch=VoicePitch.high),
    Augmentation(rate=VoiceRate.medium, pitch=VoicePitch.low),
    Augmentation(rate=VoiceRate.medium, pitch=VoicePitch.medium),
    Augmentation(rate=VoiceRate.medium, pitch=VoicePitch.high),
    Augmentation(rate=VoiceRate.xfast, pitch=VoicePitch.low),
    Augmentation(rate=VoiceRate.xfast, pitch=VoicePitch.medium),
    Augmentation(rate=VoiceRate.xfast, pitch=VoicePitch.high),
]


OUT_DIR = create_user_dir('datasets/generated/synthetic_direction_commands')

def generate_dataset():
    """Generate the dataset

    This generates the dataset using the `AudioDatasetGenerator <https://siliconlabs.github.io/mltk/docs/python_api/utils/audio_dataset_generator/index.html>`
    Python package provided by the MLTK.

    """
    with AudioDatasetGenerator(
        out_dir=OUT_DIR,
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

        # See: https://docs.aws.amazon.com/polly/latest/dg/get-started-what-next.html
        if 'AWS_ACCESS_KEY_ID' in os.environ or os.path.exists(os.path.expanduser('~/.aws')):
            generator.load_backend('aws', install_python_package=True)
            print('Loaded AWS backend')

        # See: https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/get-started-text-to-speech?pivots=programming-language-python
        if 'SPEECH_KEY' in os.environ:
            generator.load_backend('azure', install_python_package=True)
            print('Loaded Azure backend')

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