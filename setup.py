long_description = """Silicon Labs Machine Learning Toolkit (MLTK)
==============================================

__NOTICE:__  
This package is considered EXPERIMENTAL - SILICON LABS DOES NOT OFFER ANY WARRANTIES AND DISCLAIMS ALL IMPLIED WARRANTIES CONCERNING THIS SOFTWARE. 
This package is made available as a self-serve reference supported only by the on-line documentation, and community support. 
There are no Silicon Labs support services for this software at this time.


This is a Python package with command-line utilities and scripts to aid the development 
of machine learning models for Silicon Lab's embedded platforms.

See the [MLTK Overview](https://siliconlabs.github.io/mltk/docs/overview.html) for an overview of how the various features of the MLTK are used to
create machine learning models for embedded devices.

The features of this Python package include:
- [Command-line](https://siliconlabs.github.io/mltk/docs/command_line.html) - Execute all ML operations from simple command-line interface
- [Python API](https://siliconlabs.github.io/mltk/docs/python_api/python_api.html) - Execute all ML operations from a Python script
- [Model Profiler](https://siliconlabs.github.io/mltk/docs/guides/model_profiler.html) - Determine how efficient an ML model will execute on an embedded platform
- [Model Training](https://siliconlabs.github.io/mltk/docs/guides/model_training.html) - Train an ML model using [Google Tensorflow](https://www.tensorflow.org/)
- [Remote Training via SSH](./docs/guides/model_training_via_ssh.md) - Securely and seamlessly train the model on a remote "cloud" machine
- [Model Evaluation](https://siliconlabs.github.io/mltk/docs/guides/model_evaluation.html) - Evaluate a trained ML model's accuracy and other metrics
- [Model Summary](https://siliconlabs.github.io/mltk/docs/guides/model_summary.html) - Generate a summary of the model's contents
- [Model Visualization](https://siliconlabs.github.io/mltk/docs/guides/model_visualizer.html) - Interactively view the ML model's structure 
- [Model Quantization](https://siliconlabs.github.io/mltk/docs/guides/model_quantization.html) - Reduce the memory footprint of an ML model by using the [Tensorflow-Lite Converter](https://www.tensorflow.org/lite/convert)
- [Model Parameters](https://siliconlabs.github.io/mltk/docs/guides/model_parameters.html) - Embed custom parameters into the generated model file
- [Audio Utilities](https://siliconlabs.github.io/mltk/docs/audio/audio_utilities.html) - Utilities to visualize and classify real-time audio for keyword spotting
- [Python C++ Wrappers](https://siliconlabs.github.io/mltk/docs/cpp_development/wrappers/index.html) - Execute C++ libraries (including [Tensorflow-Lite Micro](https://github.com/tensorflow/tflite-micro)) from a Python interface



## Installation

```shell
# Windows
pip  install silabs-mltk

# Linux
pip3 install silabs-mltk
```

Refer to [Installation Guide](https://siliconlabs.github.io/mltk/docs/installation.html) for more details on how to install the MLTK.


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
"""


import os
import sys
import time
import re
import subprocess
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py

import mltk
from mltk.utils.path import clean_directory

sys_ver = sys.version_info
python_version = f'{sys_ver[0]}{sys_ver[1]}'
if os.name == 'nt':
    wrapper_extension = f'cp{python_version}-*'
else:
    wrapper_extension = f'cpython-{python_version}*'


curdir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')


cmdclass = {}

if os.environ.get('MLTK_NO_BUILD_WRAPPERS', None) != '1':
    try:
        from cpp.tools.setup.build_wrappers_command import BuildWrappersCommand
        cmdclass['build_ext'] = BuildWrappersCommand
    except:
        pass 

    class CustomBuildPy(build_py):
        def run(self):
            build_dir = f'{curdir}/build/lib'
            self.announce(f'Cleaning {build_dir}')
            clean_directory(build_dir)

            if 'build_ext' in cmdclass:
                # Build the MLTK C++ wrappers
                self.run_command('build_ext')
            return super().run()

    cmdclass['build_py'] = CustomBuildPy


try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class BdistWheelCommand(_bdist_wheel):
        def finalize_options(self):
            super().finalize_options()
            # Ensure the generated .whl file is specific to the current Python/OS
            # as it's dependent on the built wrappers
            self.root_is_pure = False
            # Ensure the generated .whl has a unique name
            self.build_number = f'{int(time.time())}'

    cmdclass['bdist_wheel'] = BdistWheelCommand
except:
    pass 


additional_install_dependencies = []

# If we're running Python3.7 then we also need to install pickle5
if python_version == '37':
    print('Adding pickle5 to install dependencies')
    additional_install_dependencies.append('pickle5')
# Other ensure pickle5 is NOT installed as that will break other dependencies
else:
    print('Uninstalling pickle5 (if necessary)')
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'pickle5'])


install_requires=[
    'typer<1.0',
    'pytest',
    'pytest-dependency',
    'pytest-html-reporter',
    'cmake',
    'ninja',
    'psutil',
    'pyaml<22.0',
    'tensorflow>=2.3,<3.0',
    'tensorflow_probability>=0.12.2',
    'tflite-support<0.4.2', # 'tflite-support==0.4.2 requires flatbuffers>2.0, but TF requires flatbuffers<2.0
    'protobuf>=3.18,<3.20', # The MLTK does NOT have a dependency on this, but tflite-support and tensorflow do
    'onnx',
    'onnxruntime<1.11',
    'flatbuffers<2.0', # This is required by TF
    'numpy<1.23', # Numba, which is installed by TF, has a requirement of < 1.23
    'scipy<2.0',
    'matplotlib<4.0',
    'tqdm<5.0',
    'pillow<9.0',
    'librosa<1.0',
    'joblib',
    'bincopy<18.0',
    'pyserial<4.0',
    'GPUtil<2.0',
    'patool==1.12',
    'prettytable>=2.0,<3.0'
] + additional_install_dependencies

setup_dependencies_py = os.environ.get('MLTK_SETUP_PY_DEPS', '').split('|')
package_name_re = re.compile(f'^(\w+)') # Find everything before the non-alphanumeric characters
for dep in setup_dependencies_py:
    match = package_name_re.match(dep)
    if not match:
        continue
    dep_name = match.group(1).lower()
    modified = False
    for i, req in enumerate(install_requires):
        # If the MLTK_SETUP_PY_DEPS is already an install requirement, 
        # then just replace it
        if req.lower().startswith(dep_name): 
            install_requires[i] = dep 
            print(f'Modifying install requirement: {dep}')
            modified = True 
            break
    # Otherwise add the new MLTK_SETUP_PY_DEPS to the install requirements
    if not modified:
        print(f'Adding install requirement: {dep}')
        install_requires.append(dep)


setup(
    name='silabs-mltk',
    version=mltk.__version__,
    description='This allows for developing embedded machine learning models using Tensorflow-Lite Micro',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://siliconlabs.github.io/mltk',
    author='Silicon Labs',
    license='SPDX-License-Identifier: Zlib',
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.7,<3.10',
    setup_requires=['wheel'],
    install_requires=install_requires,
    packages=find_packages(include=['mltk', 'mltk.*']),
    package_dir={'': '.'},
    package_data={ 
        'mltk.core.tflite_micro': [f'_tflite_micro_wrapper.{wrapper_extension}'],
        'mltk.core.tflite_micro.accelerators.mvp': [f'_mvp_wrapper.{wrapper_extension}'],
        'mltk.core.tflite_micro.accelerators.mvp.estimator': ['estimators_url.yaml'],
        'mltk.core.preprocess.audio.audio_feature_generator': [f'_audio_feature_generator_wrapper.{wrapper_extension}'],
        'mltk.core.tflite_model_parameters.schema': ['dictionary.fbs', 'generate_schema.sh'],
        'mltk.models.examples': [
            'audio_example1.mltk.zip', 
            'image_example1.mltk.zip', 
            'autoencoder_example.mltk.zip',
        ],
        'mltk.models.siliconlabs': [
            'fingerprint_signature_generator.mltk.zip',
            'keyword_spotting_on_off.mltk.zip',
            'keyword_spotting_on_off_v2.mltk.zip',
            'keyword_spotting_mobilenetv2.mltk.zip',
            'keyword_spotting_with_transfer_learning.mltk.zip',
            'keyword_spotting_pacman.mltk.zip',
            'rock_paper_scissors.mltk.zip'
        ],
        'mltk.models.tflite_micro': [
            'tflite_micro_speech.mltk.zip',
            'tflite_micro_magic_wand.mltk.zip'
        ],
        'mltk.models.tinyml': [
            'anomaly_detection.mltk.zip', 
            'image_classification.mltk.zip', 
            'keyword_spotting.mltk.zip', 
            'visual_wake_words.mltk.zip'
        ],
        'mltk.utils.firmware_apps': ['download_urls.yaml'],
        'mltk.utils.test_helper.data': ['*.tflite', '*.h5'],
        'mltk.utils.audio_visualizer.settings': ['config.yaml'],
        'mltk.utils.audio_visualizer.gui': ['favicon.ico'],
        'mltk.examples': ['*.md', '.ipynb'],
        'mltk.tutorials': ['*.md', '.ipynb'],
    },
    cmdclass=cmdclass,
    entry_points = {
        'console_scripts': ['mltk=mltk.cli.main:main'],
    }
)