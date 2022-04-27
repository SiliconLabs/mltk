# C++ Source Code

This directory contains all of the C++ related source files and tools used by the MLTK.

This directory has the following structure:

```
/tflite_micro_wrapper                        - Tensorflow-Lite Micro Python wrapper, this allows for executing the Tensorflow-Lite Micro interpreter from a Python script
/audio_feature_generator_wrapper             - The AudioFeatureGenerator Python wrapper, this allows for executing the spectrogram generation algorithms from a Python script
/mvp_wrapper                                 - MVP hardware accelerator Python wrapper, this allows for executing the MVP-accelerated Tensorflow-Lite Micro kernels from a Python script
/shared                                      - All of the C++ libraries and source code
/shared/apps                                 - Example applications and demos
/shared/platforms                            - Supported hardware platforms
/tools                                       - Tools used by the C++ build scripts
```

Refer to the [C++ Development Guide](https://siliconlabs.github.io/mltk/docs/cpp_development/index.html) for more details.