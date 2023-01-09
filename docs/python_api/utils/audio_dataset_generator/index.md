__NOTE:__ Refer to the [online documentation](https://siliconlabs.github.io/mltk) to properly view this file
# Audio Dataset Generator

This allows for generating a synthetic keyword audio datasets using:
- [Google Cloud Platform (GCP)](https://cloud.google.com/text-to-speech)
- [Microsoft (Azure)](https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/text-to-speech)
- [Amazon Web Services (AWS)](https://aws.amazon.com/polly)

See the [Synthetic Audio Dataset Generation](https://siliconlabs.github.io/mltk/mltk/tutorials/synthetic_audio_dataset_generation.html) tutorial for more details.


## Example Usage

The following is a snippet taken from [alexa_dataset_generator.py](https://github.com/siliconlabs/mltk/blob/master/mltk/utils/audio_dataset_generator/examples/alexa_dataset_generator.py)


```python
import tqdm
from mltk.utils.audio_dataset_generator import (
    AudioDatasetGenerator,
    Keyword,
    Augmentation,
    VoiceRate,
    VoicePitch
)


# Define the directory where the dataset will be generated
OUT_DIR = 'alexa_dataset'

# Define the keywords and corresponding aliases to generate
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

# Define the augmentations to apply the keywords
AUGMENTATIONS = [
    Augmentation(rate=VoiceRate.xslow, pitch=VoicePitch.medium),
    Augmentation(rate=VoiceRate.medium, pitch=VoicePitch.medium),
    Augmentation(rate=VoiceRate.xfast, pitch=VoicePitch.medium),
]

# Instantiate the AudioDatasetGenerator
with AudioDatasetGenerator(
    out_dir=OUT_DIR,
    n_jobs=2
) as generator:
    # Load the cloud backends
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

    # Generate the dataset (with pretty progress bars)
    print(f'Generating keywords at: {generator.out_dir}\n')
    with tqdm.tqdm(total=n_configs, desc='Overall'.rjust(10), unit='word', position=1) as pb_outer:
        for keyword, config_list in all_configurations.items():
            with tqdm.tqdm(desc=keyword.value.rjust(10), total=len(config_list), unit='word', position=0) as pb_inner:
                for config in config_list:
                    generator.generate(config, on_finished=lambda x: (pb_inner.update(1), pb_outer.update(1)))
                generator.join() # Wait for the current keyword to finish before continuing to the next

```



## API Reference

The following APIs are provided by this package:

```{eval-rst}

.. autosummary::
   :toctree: generator
   :template: custom-class-template.rst

   mltk.utils.audio_dataset_generator.AudioDatasetGenerator

.. autosummary::
   :toctree: voice
   :template: custom-class-template.rst

   mltk.utils.audio_dataset_generator.Voice

.. autosummary::
   :toctree: voice_pitch
   :template: custom-class-template.rst

   mltk.utils.audio_dataset_generator.VoicePitch

.. autosummary::
   :toctree: voice_rate
   :template: custom-class-template.rst

   mltk.utils.audio_dataset_generator.VoiceRate

.. autosummary::
   :toctree: keyword
   :template: custom-class-template.rst

   mltk.utils.audio_dataset_generator.Keyword

.. autosummary::
   :toctree: augmentation
   :template: custom-class-template.rst

   mltk.utils.audio_dataset_generator.Augmentation

.. autosummary::
   :toctree: generation_config
   :template: custom-class-template.rst

   mltk.utils.audio_dataset_generator.GenerationConfig

```



```{toctree}
:maxdepth: 1
:hidden:

./generator
./voice
./voice_pitch
./voice_rate
./keyword
./augmentation
./generation_config
```