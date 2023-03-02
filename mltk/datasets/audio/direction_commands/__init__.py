"""Direction Commands
=======================

This is a `synthetically <https://siliconlabs.github.io/mltk/mltk/tutorials/synthetic_audio_dataset_generation.html>`_ generated dataset with the keywords:

- **left**
- **right**
- **up**
- **down**
- **stop**
- **go**

The samples are 16kHz, 16-bit PCM ``.wav`` files.


.. seealso::

    - `AudioDatasetGenerator <https://siliconlabs.github.io/mltk/docs/python_api/utils/audio_dataset_generator/index.html>`_
    - `Synthetic Audio Dataset Generation Tutorial <https://siliconlabs.github.io/mltk/mltk/tutorials/synthetic_audio_dataset_generation.html>`_

"""


from .download import download, DOWNLOAD_URL, VERIFY_SHA1, CLASSES
from .generate_dataset import generate_dataset
