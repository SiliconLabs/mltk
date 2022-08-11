# Model Profiler

The model profiler application allows for running `.tflite` model files in the Tensorflow-Lite Micro interpreter with optional hardware acceleration.
The model profiler application can run on Windows/Linux via simulator _or_ on a supported embedded platform.

If a hardware simulator is used, then only accelerator cycles are profiled.  
If an embedded platform is used, then accelerator cycles and CPU cycles are profiled.


__NOTES:__  
- This application is able to be built for Windows/Linux _or_ a supported embedded target.
- Rather than build this C++ application, you may also profile models using:  
  - [Command Line](https://siliconlabs.github.io/mltk/docs/guides/model_profiler.html) - Profile a .tflite model using the `mltk profile` command line
  - [Model Profiler Utility](https://siliconlabs.github.io/mltk/docs/guides/model_profiler_utility.html) - Profile a .tflite using a standalone utility with a webpage interface


## Quick Links

- [GitHub Source](https://github.com/SiliconLabs/mltk/tree/master/cpp/shared/apps/model_profiler) - View this example's source code on Github
- [Online documentation](https://siliconlabs.github.io/mltk/docs/cpp_development/examples/model_profiler.html) - View this example's online documentation
- [Model Profiler Command Line](https://siliconlabs.github.io/mltk/docs/guides/model_profiler.html) - View the MLTK's model profiler documentation
- [Model Profiler Utility](https://siliconlabs.github.io/mltk/docs/guides/model_profiler_utility.html) - View the MTLK's model profiler stand-alone utility's documentation



## Build, Run, Debug

See the [online documentation](https://siliconlabs.github.io/mltk/docs/cpp_development/index.html) for how to build and run this application:


### Simplicity Studio

If using [Simplicity Studio](https://siliconlabs.github.io/mltk/docs/cpp_development/simplicity_studio.html) select the `MLTK - Model Profiler` Project.

### Visual Studio Code
If using [Visual Studio Code](https://siliconlabs.github.io/mltk/docs/cpp_development/vscode.html) select the `mltk_model_profiler` CMake target.

### Command-line

If using the [Command Line](https://siliconlabs.github.io/mltk/docs/cpp_development/command_line.html) select the `mltk_model_profiler` CMake target.  



## Updating the model

The application will profile _any_ quantized `.tflite` model file. 
A default model comes with the application, however, this model may be updated 
using several different methods:


### via Simplicity Studio

To replace the default model, rename the your `.tflite` file to
`1_<your model named>.tflite` and copy it into the config/tflite folder of Simplicity Studio
project. (Simplicity Studio sorts the models alphabetically in ascending order, adding `1_` 
forces the model to come first). After a new .tflite file is added to the 
project Simplicity Studio will automatically use the 
[flatbuffer converter tool](https://docs.silabs.com/gecko-platform/latest/machine-learning/tensorflow/flatbuffer-conversion)
to convert a .tflite file into a c file which is added to the project.

Refer to the online [documentation](https://docs.silabs.com/gecko-platform/latest/machine-learning/tensorflow/guide-replace-model#updating-or-replacing-the--tflite-file-in-a-project) for more details.


### via CMake

The model can also be updated when building this application from [Visual Studio Code](https://siliconlabs.github.io/mltk/docs/cpp_development/vscode.html)
or the CMake [Command Line](https://siliconlabs.github.io/mltk/docs/command_line.html).

To update the model, create/modify the file: `<mltk repo root>/user_options.cmake`
and add:

```
mltk_set(MODEL_PROFILER_MODEL <model name or path>)
```

where `<model name or path>` is the file path to your model's `.tflite` 
or the MLTK model name.

With this variable set, when the model profiler application is built the 
specified model will be built into the application.


### via update_params command

When building for an embedded target, this application supports overriding the default model built into the application.
When the application starts, it checks the end of flash memory for a `.tflite` model file. If found, the model
at the end of flash is used instead of the default model.

To write the model to flash, use the command:

```shell
mltk update_params <model name> --device
```

Refer to the command's help for more details:

```shell
mltk update_params --help
```




## Build Settings

When building this application using [Visual Studio Code](https://siliconlabs.github.io/mltk/docs/cpp_development/vscode.html) 
or the [Command Line](https://siliconlabs.github.io/mltk/docs/cpp_development/command_line.html) several options may be configured 
to control how the app is built.

To specify the settings, create/modify the file:  
`<mltk repo root>/user_options.cmake`


The following settings are supported:


### MODEL_PROFILER_MODEL

Optionally, configure the `.tflite` model to profile:

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




### TFLITE_MICRO_ACCELERATOR

Optionally, configure the target hardware accelerator:

```shell
# Use the Gecko SDK MVP TFLM kernels
mltk_set(TFLITE_MICRO_ACCELERATOR mvp)
```


### TFLITE_MICRO_ACCELERATOR_PROFILER_ENABLED

Enable/disable profiling MVP performance counters

```shell
mltk_set(TFLITE_MICRO_ACCELERATOR_PROFILER_ENABLED ON)
```


### MLTK_RUN_MODEL_FROM_RAM

If enabled, the `.tflite` model file is copied from flash to RAM
and the model executes entirely from RAM.

```shell
mltk_set(MLTK_RUN_MODEL_FROM_RAM ON)
```

__NOTE:__ To use this, the `.tflite` _must_ fit into RAM along with the normal runtime working memory.


### MLTK_RUNTIME_MEMORY_SIZE

If specified, then hardcode the tensor arena size to the given value.
If omitted, then automatically find the optimal tensor arena size.

```shell
mltk_set(MLTK_RUNTIME_MEMORY_SIZE 100000)
```


### TFLITE_MICRO_RECORDER_ENABLED

If enabled, then record each layer's input/output tensor data.

```shell
mltk_set(TFLITE_MICRO_RECORDER_ENABLED ON)
```
