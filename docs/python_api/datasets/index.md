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
   :toctree: audio/on_off
   :template: custom-module-template.rst

    mltk.datasets.audio.on_off

.. autosummary::
   :toctree: audio/yes_no
   :template: custom-module-template.rst

    mltk.datasets.audio.yes_no

.. autosummary::
   :toctree: audio/ten_digits
   :template: custom-module-template.rst

    mltk.datasets.audio.ten_digits

.. autosummary::
   :toctree: audio/hey_gecko
   :template: custom-module-template.rst

    mltk.datasets.audio.hey_gecko

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

.. autosummary::
   :toctree: audio/mit_ir_survey
   :template: custom-module-template.rst

    mltk.datasets.audio.mit_ir_survey

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
./audio/on_off
./audio/yes_no
./audio/ten_digits
./audio/ml_commons/keywords
./audio/ml_commons/voice
./audio/mit_ir_survey
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