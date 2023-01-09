# C++ Build Options

Custom CMake settings may be specified in the file:

```
<mltk repo root>/user_options.cmake
```

__NOTE:__  You must create this file if it doesn't exist


This file is included at the very beginning of the CMake build script.
Any CMake variable may be include in the file.  
Some of the more common settings include:



## MLTK_CMAKE_LOG_LEVEL

```
# Enable verbose MLTK CMake log messages
mltk_set(MLTK_CMAKE_LOG_LEVEL debug)
```


## MLTK_PLATFORM_NAME

Build for a supported embedded platform.

Refer to [Supported Hardware](../other/supported_hardware.md) for more details.

__NOTE:__ If this variable is not defined, then the host operating system (e.g. `windows` or `linux`) is automatically selected.

```shell
# Specify the embedded platform
mltk_set(MLTK_PLATFORM_NAME brd2601)
```


## TFLITE_MICRO_ACCELERATOR

Specify the Tensorflow-Lite Micro accelerated kernels to include in the build.

```shell
# Use the Gecko SDK MVP TFLM kernels
mltk_set(TFLITE_MICRO_ACCELERATOR mvp)
```

## MLTK_TARGET

Build for a specific target.  
By default, all applications are automatically included during the CMake configuration phase.
This option allows only include a specific package in the configuration.

__NOTE:__ The specified target must be found via `find_package()` or `mltk_find_package()`.

```shell
# Only include the TFLM Python wrapper in the build
mltk_set(MLTK_TARGET mltk_tflite_micro_wrapper)
```


## MODEL_PROFILER_MODEL

```shell
# Specify the path to the .tflite model file to profile
# in the mltk_model_profiler application
mltk_set(MODEL_PROFILER_MODEL ~/my_models/my_model.tflite)
```

__HINT:__  
You can also specify the path to the `.mltk.zip` model archive or just specify the MLTK model name, e.g.:

```shell
# Specify the path to the model archive
mltk_set(MODEL_PROFILER_MODEL ~/my_models/my_model.mltk.zip)

# Specify the MLTK model name
# NOTE: The model specification must be on the model search path, see:
#       https://siliconlabs.github.io/mltk/docs/guides/model_search_path.html
mltk_set(MODEL_PROFILER_MODEL image_example1)
```



## AUDIO_CLASSIFIER_MODEL

```
# Specify the path to the .tflite model file to profile
# in the mltk_audio_classifier application
mltk_set(AUDIO_CLASSIFIER_MODEL ~/my_models/my_model.tflite)
```

__HINT:__  
You can also specify the path to the `.mltk.zip` model archive or just specify the MLTK model name, e.g.:

```shell
# Specify the path to the model archive
mltk_set(AUDIO_CLASSIFIER_MODEL ~/my_models/my_model.mltk.zip)

# Specify the MLTK model name
# NOTE: The model specification must be on the model search path, see:
#       https://siliconlabs.github.io/mltk/docs/guides/model_search_path.html
mltk_set(AUDIO_CLASSIFIER_MODEL image_example1)
```


## MLTK_RUN_MODEL_FROM_RAM

In the [__mltk__/cpp/shared/apps/model_profiler](../../cpp/shared/apps/model_profiler) app, load the `.tflite` model into RAM before
running inference. This causes all models weights and filters to 
reside in RAM as well.

__NOTE:__ Bobcat only has 256K of RAM total. So the model
must be small enough to fit the model + TFLM working memory.

```shell
mltk_set(MLTK_RUN_MODEL_FROM_RAM ON)
```

