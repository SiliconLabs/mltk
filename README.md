# Silicon Labs Machine Learning Toolkit (MLTK)


```{warning}
This package is considered EXPERIMENTAL - SILICON LABS DOES NOT OFFER ANY WARRANTIES AND DISCLAIMS ALL IMPLIED WARRANTIES CONCERNING THIS SOFTWARE. 
This package is made available as a self-serve reference supported only by the on-line documentation, and community support. 
There are no Silicon Labs support services for this software at this time.
```

<a href="https://siliconlabs.github.io/mltk" target="_blank">![version](https://img.shields.io/badge/MLTK%20Version-0.20.0-red?style=for-the-badge)</a>
<a href="https://pypi.org/project/silabs-mltk" target="_blank">![PyPI - Python Version](https://img.shields.io/pypi/pyversions/silabs-mltk?style=for-the-badge)</a>
<a href="https://github.com/SiliconLabs/gecko_sdk/tree/v4.3.3" target="_blank">![gsdk](https://img.shields.io/badge/Gecko%20SDK-4.3.3-green?style=for-the-badge)</a>
<a href="https://github.com/tensorflow/tflite-micro/tree/7005d60ede074762f01c1d0fd24ec58240af89b5" target="_blank">![tflm](https://img.shields.io/badge/Tensorflow--Lite%20Micro-March%202024-orange?style=for-the-badge)</a>
<a href="https://www.tensorflow.org/api_docs" target="_blank">![tf](https://img.shields.io/badge/Tensorflow-2.16-yellow?style=for-the-badge)</a>

This is a Python package with command-line utilities and scripts to aid the development 
of machine learning models for Silicon Lab's embedded platforms.

The features of this Python package include:
- [Command-line](./docs/command_line/index.md) - Execute all ML operations from a simple command-line interface
- [Python API](./docs/python_api/index.md) - Execute all ML operations from a Python script
- [Model Profiler](./docs/guides/model_profiler.md) - Determine how efficiently an ML model will execute on an embedded platform
- [Model Training](./docs/guides/model_training.md) - Train an ML model using [Google Tensorflow](https://www.tensorflow.org/)
- [Remote Training via SSH](./docs/guides/model_training_via_ssh.md) - Securely and seamlessly train the model on a remote "cloud" machine
- [Model Training Monitor](./docs/guides/model_training_monitor.md) - Monitor/profile the training of a model using [Tensorboard](https://www.tensorflow.org/tensorboard)
- [Model Evaluation](./docs/guides/model_evaluation.md) - Evaluate a trained ML model's accuracy and other metrics
- [Model Summary](./docs/guides/model_summary.md) - Generate a summary of the model's contents
- [Model Visualization](./docs/guides/model_visualizer.md) - Interactively view the ML model's structure 
- [Model Quantization](./docs/guides/model_quantization.md) - Reduce the memory footprint of an ML model by using the [Tensorflow-Lite Converter](https://www.tensorflow.org/lite/convert)
- [Model Parameters](./docs/guides/model_parameters.md) - Embed custom parameters into the generated model file
- [Audio Feature Generator](./docs/audio/audio_feature_generator.md) - Library and tools to convert streaming audio into spectrograms
- [Audio Utilities](./docs/audio/audio_utilities.md) - Utilities to aid the development of audio classification models
- [C++ Python Wrappers](./docs/cpp_development/wrappers/index.md) - Enable sharing source code between embedded targets and model training scripts


Refer to [Why MLTK?](./docs/why_mltk.md) for more details on the benefits of using the MLTK.

```{hint} 
Just want to quickly profile a model to see how fast it can run on an embedded target?  
See the [Model Profiler Utility](./docs/guides/model_profiler_utility.md)
```


## Overview

```{eval-rst}
.. raw:: html

   <iframe src="./_static/overview/index.html" height="100%" width="100%" frameborder="0" class="slideshow-iframe" allowfullscreen></iframe>
```


## Installation

Install the pre-built Python package:

```{eval-rst}
.. tab-set::

   .. tab-item:: Windows

      .. code-block:: shell

         pip  install silabs-mltk

   .. tab-item:: Linux

      .. code-block:: shell

         pip3 install silabs-mltk

```


Or, build and install the Python package from [Github](https://github.com/siliconlabs/mltk):

```{eval-rst}
.. tab-set::

   .. tab-item:: Windows

      .. code-block:: shell

         pip  install git+https://github.com/siliconlabs/mltk.git

   .. tab-item:: Linux

      .. code-block:: shell

         pip3 install git+https://github.com/siliconlabs/mltk.git
```

Refer to the [Installation Guide](./docs/installation.md) for more details on how to install the MLTK.


## Other Information

- [Frequently Asked Questions](./docs/faq/index.md)
- [Model Profiler Utility](./docs/guides/model_profiler_utility.md)
- [Quick Reference](./docs/other/quick_reference.md)
- [Settings File](./docs/other/settings_file.md)
- [Model Specification](./docs/guides/model_specification.md)
- [Model Archive File](./docs/guides/model_archive.md)
- [Model Search Path](./docs/guides/model_search_path.md)
- [Environment Variables](./docs/other/environment_variables.md)
- [C++ Development](./docs/cpp_development/index.md)
- [Ask a Question](https://github.com/SiliconLabs/mltk/issues)

## License

SPDX-License-Identifier: Zlib

The licensor of this software is Silicon Laboratories Inc.

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.