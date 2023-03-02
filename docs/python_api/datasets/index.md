__NOTE:__ Refer to the [online documentation](https://siliconlabs.github.io/mltk) to properly view this file
# Reference Datasets

The MLTK comes with datasets that are used by the [reference models](../models/index.md).

The source code for these datasets may be found on Github: [https://github.com/siliconlabs/mltk/tree/master/mltk/datasets](https://github.com/siliconlabs/mltk/tree/master/mltk/datasets).



## Audio Datasets

```{eval-rst}

.. autosummary::
   :toctree: audio/speech_commands_v2
   :template: custom-module-template.rst

    mltk.datasets.audio.speech_commands.speech_commands_v2

.. autosummary::
   :toctree: audio/direction_commands
   :template: custom-module-template.rst

    mltk.datasets.audio.direction_commands

.. autosummary::
   :toctree: audio/ml_commons/keywords
   :template: custom-module-template.rst

    mltk.datasets.audio.mlcommons.ml_commons_keywords

.. autosummary::
   :toctree: audio/ml_commons/voice
   :template: custom-module-template.rst

    mltk.datasets.audio.mlcommons.ml_commons_voice

.. autosummary::
   :toctree: audio/background_noise/ambient
   :template: custom-module-template.rst

    mltk.datasets.audio.background_noise.ambient

.. autosummary::
   :toctree: audio/background_noise/brd2601
   :template: custom-module-template.rst

    mltk.datasets.audio.background_noise.brd2601

.. autosummary::
   :toctree: audio/background_noise/esc50
   :template: custom-module-template.rst

    mltk.datasets.audio.background_noise.esc50

```

## Image Datasets

```{eval-rst}

.. autosummary::
   :toctree: image/rock_paper_scissors_v1
   :template: custom-module-template.rst

    mltk.datasets.image.rock_paper_scissors_v1

.. autosummary::
   :toctree: image/rock_paper_scissors_v2
   :template: custom-module-template.rst

    mltk.datasets.image.rock_paper_scissors_v2

.. autosummary::
   :toctree: image/mnist
   :template: custom-module-template.rst

    mltk.datasets.image.mnist

.. autosummary::
   :toctree: image/cifar10
   :template: custom-module-template.rst

    mltk.datasets.image.cifar10

.. autosummary::
   :toctree: image/fashion_mnist
   :template: custom-module-template.rst

    mltk.datasets.image.fashion_mnist

```



## Accelerometer Datasets


```{eval-rst}

.. autosummary::
   :toctree: accelerometer/tflm_magic_wand
   :template: custom-module-template.rst

    mltk.datasets.accelerometer.tflm_magic_wand
```


```{toctree}
:maxdepth: 1
:hidden:

./audio/speech_commands_v2
./audio/direction_commands
./audio/ml_commons/keywords
./audio/ml_commons/voice
./audio/background_noise/ambient
./audio/background_noise/brd2601
./audio/background_noise/esc50
./image/rock_paper_scissors_v1
./image/rock_paper_scissors_v2
./image/mnist
./image/cifar10
./image/fashion_mnist
./accelerometer/tflm_magic_wand

```