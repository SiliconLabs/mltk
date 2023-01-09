"""
This script generates a synthetic "alexa" dataset.

See the corresponding tutorial for more details:
https://siliconlabs.github.io/mltk/mltk/tutorials/synthetic_audio_dataset_generation.html

"""

# Import the necessary Python packages
import os
import json
import tqdm
import tempfile
from mltk.utils.audio_dataset_generator import (
    AudioDatasetGenerator,
    Keyword,
    Augmentation,
    VoiceRate,
    VoicePitch
)


# NOTE: The following credentials are provided as an example.
#       You must generate your own credentials to run this example

###################################################################################################
# Configure your Azure credentials
# See: https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/get-started-text-to-speech?pivots=programming-language-python
os.environ['SPEECH_KEY'] = 'e8699507e7c04a4cb8afdba62986987c'
os.environ['SPEECH_REGION'] = 'westus2'


###################################################################################################
# Configure your Google credentials
# See: https://codelabs.developers.google.com/codelabs/cloud-text-speech-python3

# NOTE: The "serivce account" JSON was copied into this Python script for demonstration purposes.
#       You could also just set GOOGLE_APPLICATION_CREDENTIALS to point to your service account .json file and
#       and remove the following.
#       If you do copy and paste into this script, be sure that the "private_key" is on a single line, e.g.:
#
#       "private_key_id": ...,
#       "private_key": "-----BEGIN PRIVATE KEY---- ....",
#       "client_email":  ...,
#
#       NOT:
#
#       "private_key": "-----BEGIN PRIVATE KEY--- ...
#       NEB6Y5ZODG2DYJmM+JdAHcNaPRD9/hAMRG3jl2jisVZO ...
#       03aEXJYOEWTbLWfPYxpNQyz4wKBgQDD+yVYWCrbXEECn ...
#       ... -----END PRIVATE KEY-----\n",
#       "client_email": ...,
#       "client_id": ...,
#
gcp_service_account_json_path = f'{tempfile.gettempdir()}/gcp_key.json'
gcp_service_account_json = """
{
  "type": "service_account",
  "project_id": "keyword-generator-367517",
  "private_key_id": "2af4482dd0beb3c0b2e54739a9968b43bacecf84",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCuQ4FpO6IlIB78\nmHYRHb1Ei2PCEgtthRlXbQwE6RsWppTtopQVpLSBXs30FRarpd6d4hgqeL46gC2d\nCRHH8oMrgKMB4pAGzHCEfJd/XjKckNsyIPLTqGBjAu3pT/wMukIHdYiYzDD6qjr3\nuGghP8HkT1gXgcGdFkpLWVoj9b3M6b5/3cVgBthciycCCYkHqnFOn6MTEe6OFMPZ\nWXY3FrEwyWjIWIIIvPbQaNoIJs92Gb+FFGsG2Ta63TgsZmBvVHjtd3A98EwmvwSz\nbIPXjqh5qLh3YCHdGT42MqBXrInN11kMyOC56A2Ic4mvrQ3I8oAPOs2L6ugLwX9J\nS6Sq5JW1AgMBAAECggEAT7pS2vKNnK61fpvCaNJSZangWkonMFRU48rgVN7RpetQ\n9+gKGFziuM3HLIT5ek7JKzLmG4higCFkvRQJLpGlsaGI8rPVcUbXs8XNCljujvM3\nVhf9ARln/+S3NKeDic8tpnv/oujI/+YiVHPqMEwbSXmDtD2Jd3VbSF34/7rOu5Dz\n56bGmBbNEB6Y5ZODG2DYJmM+JdAHcNaPRD9/hAMRG3jl2jisVZOgrleNelkZnrPe\n9t0uWqIv5EJItoVBZd+EzADFfjfTDrKfWv1QixeMiak1aTbs5bHKNK5ecYFFMpms\nCIVgp3wRxq7nFrJkTnWdJzeAFjQw4CKWLmN4xc2FgQKBgQDjobnxjGO7GQ1pfEiQ\nVsSuWJiXy63trU6jwrrhR1B9XUPh6VivH2dZ4lPfPywER9LX6oMTn6AIzihTPq1I\n19eskH0H6hwBw2yDzWgHZRMHB9xs5Ys+HiBKWrZ9NW77uWH1D9g/EJcN6A2ZL2ig\nK03aEXJYOEWTbLWfPYxpNQyz4wKBgQDD+yVYWCrbXEECnA0fohw8wIRo6dS6G84M\nMCzkr0YooxPb8zrIIm+mv7PAcCElaSz4LZbC2Hcb1mvV9p6o2IEUHqNgabWLFWiD\ng7CC7rm4qEE87p5U4oBUhPCiuZpA3UeAqBhxMWd1oXw5rJVXenNe+7G4JZKxERU/\nQIf7cw6zhwKBgBy5dctjWdpsSOL8yfNc36jYiTjufN43Nms30XlIFIIdWMmTNpuy\nrMoM42SShi1sGtEgSLYbOIij6zbF+/vrMM4X1Y9AHZSjYngnXW9Bc+s5NLmRJccK\n6iw30jtumLivJgtUmocqwsUAeWbRMrSzgjl4ZiN3xl/aIfkcPTGxfg7dAoGAVz6b\njmuZkJPOIRJFSVrKhUUS7P2DhOJR5N0hbyCT9A09DwKFnYiu+aWHqNiB+PyMV2M8\nJTtmMs9OrC6gzPus4r8M7iPA/Myn/TwHvRH3PbwxZqW3eIRoqrePxHEpuUyIwz6R\nuvpKW3RrL+WjihDqAVO89wRK/GZldgYNQyQiXEsCgYEAj+8nsq1UGod7SqfPiA/n\n3Wur4A+UYT8/nuaTK2WW/GTBC+eDDjRE1lZ3f/UQGTSXLSV7T1mw4a7EKrkFl36P\nLnVeFBTB3UCd8JJ0LPBtOqru9I8ns+a4FqOPMljoYElGtyT1Oy+vxfwYA7cmRz/d\n49bE21meuV3pRV1QWrrteEM=\n-----END PRIVATE KEY-----\n",
  "client_email": "my-tts-sa@keyword-generator-367517.iam.gserviceaccount.com",
  "client_id": "118369428326213488735",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/my-tts-sa%40keyword-generator-367517.iam.gserviceaccount.com"
}
"""
gcp_service_account_json = gcp_service_account_json.strip().replace(',\n', ',').replace('\n', '\\n').replace('{\\n', '{\n').replace('\\n}', '\n}')


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = gcp_service_account_json_path
with open(os.environ['GOOGLE_APPLICATION_CREDENTIALS'], 'r') as f:
    credentials = json.load(f)
os.environ['PROJECT_ID'] = credentials['project_id']


###################################################################################################
# Configure your AWS credentials
# See: https://docs.aws.amazon.com/polly/latest/dg/get-started-what-next.html
os.environ['AWS_ACCESS_KEY_ID'] = 'AKIATZWWZR5TWBUNF6IX'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'v0IRHPUGeNwj1CA7saVduF1uxW84bgkzQpOWLfdr'
os.environ['AWS_DEFAULT_REGION'] = 'us-west-2'




###################################################################################################
# Define the directory where the dataset will be generated
OUT_DIR = f'{tempfile.gettempdir()}/alexa_dataset'.replace('\\', '/')


###################################################################################################
# Define the keywords and corresponding aliases to generate
# For the _unknown_ class (e.g. negative class), we want words that sound similar to "alexa".
# NOTE: If the base word starts with an underscore, it is not included in the generation list.
# So the generation list will be:
# alexa, ehlexa, eelexa, aalexa
# ah, aag, a, o, uh, ...
#
# The dataset will have the directory structure:
# $TEMP/alexa_dataset/alexa/sample1.wav
# $TEMP/alexa_dataset/alexa/sample2.wav
# $TEMP/alexa_dataset/alexa/...
# $TEMP/alexa_dataset/_unknown_/sample1.wav
# $TEMP/alexa_dataset/_unknown_/sample2.wav
# $TEMP/alexa_dataset/_unknown_/...
KEYWORDS = [
    Keyword('alexa',
        max_count=100, # In practice, the max count should be much larger (e.g. 10000)
        aliases=('ehlexa', 'eelexa', 'aalexa')
    ),
    Keyword('_unknown_',
        max_count=200, # In practice, the max count should be much larger (e.g. 20000)
        aliases=(
        'ah', 'aah', 'a', 'o', 'uh', 'ee', 'aww', 'ala',
        'alex', 'lex', 'lexa', 'lexus', 'alexus', 'exus', 'exa',
        'alert', 'alec', 'alef', 'alee', 'ales', 'ale',
        'aleph', 'alefs', 'alevin', 'alegar', 'alexia',
        'alexin', 'alexine', 'alencon', 'alexias',
        'aleuron', 'alembic', 'alice', 'aleeyah'
    ))
]


###################################################################################################
# Define the augmentations to apply the keywords
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


###################################################################################################
# Instantiate the AudioDatasetGenerator
with AudioDatasetGenerator(
    out_dir=OUT_DIR,
    n_jobs=8 # We want to generate the keywords across 8 parallel jobs
) as generator:
    # Load the cloud backends, installing the Python packages if necessary
    generator.load_backend('aws', install_python_package=True)
    generator.load_backend('gcp', install_python_package=True)
    generator.load_backend('azure', install_python_package=True)

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

