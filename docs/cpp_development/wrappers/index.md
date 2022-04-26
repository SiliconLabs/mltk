# C++ Python Wrappers

The MLTK supports [C++ development](../index.md) and comes with several C++ Python wrappers.  

C++ Python wrappers are C++ libraries that have an additional interface which enables them 
to be invoked from a Python script. This allows for sharing source code between embedded targets and host model training scripts.

The Python wrappers use [PyBind11](https://pybind11.readthedocs.io/en/stable/index.html) to manage
converting data between Python and C++.

The source code for the wrappers may be found on Github at [__mltk__/cpp](../../../cpp)

The following wrappers are available:  
- [Audio Feature Generator](./audio_feature_generator_wrapper.md) - Allows for sharing spectrogram generation algorithms between model training scripts and embedded targets
- [Tensorflow-Lite Micro](./tflite_micro_wrapper.md) - Allows for running Tensorflow-Lite Micro interpreter from Python
- [MVP Kernels](./mvp_wrapper.md) - Allows for running MVP hardware accelerated Tensorflow-Lite Micro kernels from Python

```{eval-rst}
.. toctree::
   :maxdepth: 1
   :hidden:

   ./audio_feature_generator_wrapper
   ./tflite_micro_wrapper
   ./mvp_wrapper
```