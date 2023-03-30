__NOTE:__ Refer to the [online documentation](https://siliconlabs.github.io/mltk) to properly view this file

Installation
=================

The MLTK supports three modes of installation:  
- [Standard Python Package](#standard-python-package) - Use the MLTK like any other package in your local Python3 environment
- [Google Colab](#google-colab) - Run the MLTK in the [Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb) cloud servers. This allows for running the MLTK _without_ installing it locally
- [Local Development](#local-development) - Locally build the MLTK C++ [Python wrappers](https://siliconlabs.github.io/mltk/docs/cpp_development/wrappers/index.html) and [examples](https://siliconlabs.github.io/mltk/docs/cpp_development/examples/index.html) from source


```{note} 
[Python3.7, Python3.8, Python3.9, Python3.10](https://www.python.org/downloads/) is required
```



## Standard Python Package

This describes how to install the MLTK Python package into your Python3 environment.  

```{note} 
- Before installing, you must have [Python3.7, 3.8, 3.9, 3.10](https://www.python.org/downloads/) installed on your computer
- Installing the MLTK will also install Google [Tensorflow](https://www.tensorflow.org/install) into your Python environment,
  if your computer has an NVidia GPU, then ensure the proper drivers are [installed](https://www.tensorflow.org/install/gpu)
- If you're using Windows, be sure to install the [Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads) which is required by [Tensorflow](https://www.tensorflow.org/install/pip)

```




1 ) __Optionally__ create and activate a Python [virtual environment](https://docs.python.org/3/tutorial/venv.html): 


This step is __highly recommended__ as the MLTK installs other dependencies like Tensorflow into the Python environment.


```{eval-rst}
.. tab-set::

   .. tab-item:: Windows

      .. code-block:: shell

         python  -m venv mltk_pyvenv
         .\mltk_pyvenv\Scripts\activate.bat

   .. tab-item:: Linux

      .. code-block:: shell

         python3 -m venv mltk_pyvenv
         source ./mltk_pyvenv/bin/activate
```

2 ) Install the MLTK Python package via [pip](https://pip.pypa.io/):  

  This installs the pre-built Python package. This is the easiest and fastest approach to installing the MLTK.  
  However, the package may not be up-to-date with the [Github repository](https://github.com/siliconlabs/mltk).

  ```{eval-rst}
.. tab-set::

   .. tab-item:: Windows

      .. code-block:: shell

         pip  install silabs-mltk[full] --upgrade

   .. tab-item:: Linux

      .. code-block:: shell
      
         pip3 install silabs-mltk[full] --upgrade
  ```

  __OR__

  This builds and installs the Python package from the [Github repository](https://github.com/siliconlabs/mltk). This may take longer
  to install but will use the most up-to-date source code.

  ```{eval-rst}
.. tab-set::

   .. tab-item:: Windows

      .. code-block:: shell

         pip  install git+https://github.com/siliconlabs/mltk.git

   .. tab-item:: Linux

      .. code-block:: shell
      
         pip3 install git+https://github.com/siliconlabs/mltk.git
  ```

  __NOTE:__ The `[full]` part of the command is _optional_. This will install additional dependencies used by some the the MLTK commands.
  Omitting this from the command will speedup installation but may cause some of the commands like `classify_audio`, `view`, `tensorboard` 
  to require additional install step.


  After the command completes, the MLTK should be available to the current Python environment.  
  You can verify by issuing the command:  

  ```shell
  mltk --help
  ```


See the [Command-Line Guide](./command_line/index.md) for more details on how to use the command-line. 

You can also import the MLTK via Python script, e.g.:

```python
from mltk.core import profile_model

profile_model('~/my_model.tflite')
```

See the [API Examples](./examples.md) for more details on how to use the MLTK [Python API](./python_api/index.md).


### Update Python Package

If the MLTK Python package has already been installed, you may update to the latest MLTK by running the command:

```{eval-rst}
.. tab-set::

   .. tab-item:: Windows

      .. code-block:: shell

         pip  install silabs-mltk[full] --upgrade

   .. tab-item:: Linux

      .. code-block:: shell

         pip3 install silabs-mltk[full] --upgrade
```

Alternatively, you can update to a specific version with:

```{eval-rst}
.. tab-set::

   .. tab-item:: Windows

      .. code-block:: shell

         pip  install silabs-mltk[full]==0.16.0

   .. tab-item:: Linux

      .. code-block:: shell

         pip3 install silabs-mltk[full]==0.16.0
```

and replace `0.16.0` with the desired version.



## Google Colab

Google offers it own _free_ Cloud servers for model training, [Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb) (a.k.a. Colab).  
This is very useful as you can leverage Google's cloud servers and GPUs for training your model.
The following describes how to install the MLTK into a Colab notebook.

1 ) Create a Google Account (if necessary)  
    Go to the [Google Signup](https://accounts.google.com/signup) page.  
    __NOTE:__ Click the __Use my current email address instead__ button to use your existing email instead of creating a gmail email address.

2 ) Refer to the [Colab Basic Features Overview](https://colab.research.google.com/notebooks/basic_features_overview.ipynb) to get a basic idea of how notebooks work  
3 ) Create a new Colab notebook  
4 ) Create a Python code cell and copy & paste the following into the cell

```shell
!pip install --upgrade silabs-mltk
```

5 ) Execute the cell  
    Once the cell executes, the MLTK will be installed.
    You may import and use the MLTK package as normal from this point on inside the notebook


## Local Development

The MLTK can also be installed for local development. In this mode, the Python C++ wrappers are built from source.  
Additionally, a new Python virtual environment is created specifically for the MLTK.

```{note}
Before installing, you must have [Python3.7, 3.8, 3.9, 3.10](https://www.python.org/downloads/) installed on your computer
```

1 ) Clone the MLTK GIT repository

```shell
git clone https://github.com/siliconlabs/mltk
```

2 ) Run the install script at the root of the repository


```{eval-rst}
.. tab-set::

   .. tab-item:: Windows

      .. code-block:: shell

         cd mltk
         python  .\install_mltk.py

   .. tab-item:: Linux

      .. code-block:: shell

         cd mltk
         python3 ./install_mltk.py
```

The install script will:
1. Create a python virtual environment at `<mltk root>/.venv`
2. Install the MLTK Python package for local development into Python virtual environment:
   ```shell
   pip install -e .
   ```


```{seealso}  
- [C++ Development](./cpp_development/index.md) - Describes how to build and run MLTK C++ applications
```