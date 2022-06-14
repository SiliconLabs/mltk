# Image Classifier

This application allows for classifying images retrieved from an [ArduCAM](https://www.arducam.com/product/arducam-2mp-spi-camera-b0067-arduino/) camera.
The results of the image classification are printed to the serial terminal.

Optionally, images from the camera can be dumped to the local PC via `mltk classify_image` command.

__NOTE:__ This application _only_ supports running on supported embedded targets (Windows/Linux is not supported).


## Quick Links

- [GitHub Source](https://github.com/SiliconLabs/mltk/tree/master/cpp/shared/apps/image_classifier) - View this example's source code on Github
- [Hardware Setup](#hardware-setup) - View the required hardware setup for this example
- [Online documentation](https://siliconlabs.github.io/mltk/docs/cpp_development/examples/image_classifier.html) - View this example's online documentation
- [Image Classification Tutorial](https://siliconlabs.github.io/mltk/mltk/tutorials/image_classification.html) - View this example's associated tutorial
- [Arducam Camera Module](https://www.arducam.com/product/arducam-2mp-spi-camera-b0067-arduino/) - View the camera module's product page



## Video

A demo video of this application may be found here:  
[![Rock Paper Scissors Demo](https://img.youtube.com/vi/hIfGOc9ST50/0.jpg)](https://www.youtube.com/watch?v=hIfGOc9ST50)


## Behavior

The application executes the following loop:

1. Read image from [ArduCAM](https://www.arducam.com/product/arducam-2mp-spi-camera-b0067-arduino/) camera
2. Optionally "normalize" the based on the parameters embedded in the classification `.tflite` model file (see the "Model Input Normalization" section below)
3. Give normalized image [Tensorflow-Lite Micro](https://github.com/tensorflow/tflite-micro) Interpreter
4. Process interpreter results with optional averaging and thresholding. 
   If a classification result has a high enough probability then it is considered "detected"
   In this case, print the detection result to the serial console and turn the green LED on.  
   If the probability is lower than the detection threshold, then turn the red LED on
   which indicates that there is activity but no detection.

__NOTE:__ Class labels that start with an underscore (e.g. `_unknown_`) are ignored from detection.


## Classification Model

This application expects an "image classification" model. 

### Model Input

The model input should have the shape:  `<image height>  x <image width> x <image channels>`  
where `<image channels>` should be either one (i.e. 8-bit grayscale) or three (i.e. 8-bit RGB).

The datatype should be either `int8` or `float32`.


### Model Input Normalization

The application also supports "normalizing" the input. 

If the `samplewise_norm.rescale` [model parameter](https://siliconlabs.github.io/mltk/docs/guides/model_parameters.html#imagedatasetmixin)
is given, then each element in the image is multiplied by this scaling factor, i.e.:

```
model_input_tensor = img * samplewise_norm.rescale
```

If the `samplewise_norm.mean_and_std` [model parameter](https://siliconlabs.github.io/mltk/docs/guides/model_parameters.html#imagedatasetmixin)
is given, then each element in the image is centered about its mean and scaled by its standard deviation, i.e.:

```
model_input_tensor = (img  - mean(img)) / std(img)
```


In both these cases, the model input data type must be `float32`.

If the model input data type is `int8`, then the image data type is automatically converted to from `uint8` to `int8`, i.e.:

```
model_input_tensor = (int8)(img - 128)
```

If the model input data type is `float32` and no normalization is done, then the image data type is automatically converted to from `uint8` to `float32`, i.e.:

```
model_input_tensor = (float)img
```


### Model Output

The model output should have the shape `1 x <number of classes>`  
where `<number of classes>` should be the number of classes that the model is able to detect.

The datatype should be either `int8` or `float32`  
__NOTE:__ TFLM expects the model input and output to have the same data type.



## Updating the model

The application will run _any_ quantized image classification `.tflite` model file. 
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


#### via classify_image Command

Alternatively, using the `mltk classify_image <model path> --app none`
command program the `.tflite` model to the end of the device's flash.
On startup, the application will detect the new model and use that instead
of the model built into the firmware.

__NOTE:__ The `--app none` option tells the command to _not_ update the image_classifier application and only program the model file.


### via CMake

The model can also be updated when building this application from [Visual Studio Code](https://siliconlabs.github.io/mltk/docs/cpp_development/vscode.html)
or the CMake [Command Line](https://siliconlabs.github.io/mltk/docs/command_line.html).

To update the model, create/modify the file: `<mltk repo root>/user_options.cmake`
and add:

```shell
mltk_set(IMAGE_CLASSIFIER_MODEL <model name or path>)
```

where `<model name or path>` is the file path to your model's `.tflite` 
or the MLTK model name.

With this variable set, when the image_classifier application is built the 
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


## Hardware Setup

To run this application, an [ArduCAM](https://www.arducam.com/product/arducam-2mp-spi-camera-b0067-arduino/) module is required.
This camera must be connected to the SPI and I2C peripheral of the embedded target.

The following default pin mappings are used by:  
-  __BRD2601__
-  __BRD4166__

| ArduCAM Pin | Board Expansion Header Pin |
| ----------- | -------------------------- |
| GND         | 1                          |
| VCC         | 18                         |
| CS          | 10                         |
| MOSI        | 4                          |
| MISO        | 6                          |
| SCK         | 8                          |
| SDA         | 16                         |
| SCL         | 15                         |


## Build, Run, Debug

See the [online documentation](https://siliconlabs.github.io/mltk/docs/cpp_development/index.html) for how to build and run this application:


### Simplicity Studio

If using [Simplicity Studio](https://siliconlabs.github.io/mltk/docs/cpp_development/simplicity_studio.html) select the `MLTK - Image Classifier` Project.

### Visual Studio Code
If using [Visual Studio Code](https://siliconlabs.github.io/mltk/docs/cpp_development/vscode.html) select the `mltk_image_classifier` CMake target.

### Command-line

If using the [Command Line](https://siliconlabs.github.io/mltk/docs/cpp_development/command_line.html) select the `mltk_image_classifier` CMake target.  



## Dumping images to PC

This application works with the MLTK command:

```shell
mltk classify_image --help
```

Using this command, you can dump images to the local PC.

For example:

```shell
mltk classify_image rock_paper_scissors --dump-images --dump-threshold 0.1
```

which will display the images from the camera and also save sufficiently different images as a .jpg file.



## Build Settings

When building this application using [Visual Studio Code](https://siliconlabs.github.io/mltk/docs/cpp_development/vscode.html) 
or the [Command Line](https://siliconlabs.github.io/mltk/docs/cpp_development/command_line.html) several options may be configured 
to control how the app is built.

To specify the settings, create/modify the file:  
`<mltk repo root>/user_options.cmake`



The following settings are supported:


### IMAGE_CLASSIFIER_MODEL

Optionally, configure the `.tflite` model to profile:

```shell
# Specify the path to the .tflite model file to use
# by the mltk_image_classifier application
mltk_set(IMAGE_CLASSIFIER_MODEL ~/my_models/my_model.tflite)
```

__HINT:__  
You can also specify the path to the `.mltk.zip` model archive or just specify the MLTK model name, e.g.:

```shell
# Specify the path to the model archive
mltk_set(IMAGE_CLASSIFIER_MODEL ~/my_models/my_model.mltk.zip)

# Specify the MLTK model name
# NOTE: The model specification must be on the model search path, see:
#       https://siliconlabs.github.io/mltk/docs/guides/model_search_path.html
mltk_set(IMAGE_CLASSIFIER_MODEL rock_paper_scissors)
```

### TFLITE_MICRO_ACCELERATOR

Optionally, configure the target hardware accelerator:

```shell
# Use the Gecko SDK MVP TFLM kernels
mltk_set(TFLITE_MICRO_ACCELERATOR mvp)
```
