# BLE Audio Classifier

This application uses TensorFlow Lite for Microcontrollers to run audio
classification machine learning models to classify words from audio data
recorded from a microphone. The detection is visualized using the LED's on the
board and the classification results are written to the VCOM serial port.
Additionally, the classification results are transmitted to a connected 
Bluetooth Low Energy (BLE) client.

__NOTE:__  Currently, this application only supports the [BRD2601](https://www.silabs.com/development-tools/wireless/efr32xg24-dev-kit) embedded platform


## Quick Links

- [GitHub Source](https://github.com/SiliconLabs/mltk/tree/master/cpp/shared/apps/ble_audio_classifier) - View this example's source code on Github
- [Online documentation](https://siliconlabs.github.io/mltk/docs/cpp_development/examples/ble_audio_classifier.html) - View this example's online documentation
- [Keyword Spotting Pac-Man Tutorial](https://siliconlabs.github.io/mltk/mltk/tutorials/keyword_spotting_pacman.html) - View this example's associated tutorial
- [Pac-Man Live Demo](https://mltk-pacman.web.app) - Play Pac-Man using the keywords: Left, Right, Up, Down



## Pac-Man Demo

A webpage demo has been created to work with this example application (live demo [here](https://mltk-pacman.web.app)).
The webpage connects to the development board via Bluetooth Low Energy (BLE).
Once connected, this application uses machine learning to detect the keywords:  
- Left
- Right
- Up
- Down
- Stop
- Go

When a keyword is detected, the result is sent to the webpage via BLE which then
moves the Pac-Man on the webpage accordingly.

The source code for the webpage may be here at: [__mltk repo__/cpp/shared/apps/ble_audio_classifier/web/pacman](https://github.com/SiliconLabs/mltk/tree/master/cpp/shared/apps/ble_audio_classifier/web/pacman).


## Behavior

Upon startup, the application begins advertising via Bluetooth Low Energy (BLE)
the service:  
- __Device Name:__ `MLTK KWS`
- __Service UUID:__ `c20ffe90-4ed4-46b9-8f6c-ec143fce3e4e`

Which has one characteristic with the properties: `Read` and `Notify`.
The contents of this characteristic is a string with the format:  
```
<detect-class-id>,<confidence>
```

Where:  
- __detect class id__ - Is the class ID of the detected keyword
- __confidence__ - Is the probability score of the detected keyword as a uint8 (i.e. 255 = highest confidence that the correct keyword was detected)

The BLE client is notified whenever a keyword is detected.

__NOTE:__ Keyword detection is only active when a BLE client is connected.


Additionally, the application is using two LEDs to show detection and activity and it is
printing detection results and debug log output on the VCOM serial port. In the
application configuration file called ble_audio_classifier_config.h the user can
select which LED to use for activity and which LED to use for detection. By
default the detection LED is green/led1 and the activity LED is red/led0.

At a regular interval the application will perform an inference and the result
will be processed to find the average score for each class in the current
window. If the top result score is higher than a detection threshold then a
detection is triggered and the detection LED (green) will light up for about 750
ms.

Once the detection LED turns off the application goes back to responding to the
input data. If the change in model output is greater than a configurable
sensitivity threshold, then the activity LED (red) will blink for about 500 ms.

The activity LED indicates that audio has been detected on the input and the
model output is changing, but no clear classification was made.

In audio classification, it is common to have some results that map to silence or
unknown. These results are something that we usually want to ignore. This is
being filtered out in the audio classifier application based on the label text.
By default, any labels that start with an underscore are ignored when processing
results. This behavior can be disabled in the application configuration file.

## Updating the model  

The default model used in this application is called `keyword_spotting_pacman.tflite`
and is able to classify audio into 6 different classes labeled "left", "right", "up", "down", "stop", "go". The source for the model can be found here: 
[https://github.com/siliconlabs/mltk/blob/master/mltk/models/siliconlabs/keyword_spotting_pacman.py](https://github.com/siliconlabs/mltk/blob/master/mltk/models/siliconlabs/keyword_spotting_pacman.py)


The application is designed to work with an audio classification model created
using the Silicon Labs Machine Learning Toolkit
([MLTK](https://siliconlabs.github.io/mltk/mltk/tutorials/keyword_spotting_pacman.html)). 
Use the MLTK to train a new audio classifier model and replace the model inside this example with the new audio
classification model. 

### via Simplicity Studio

To replace the default model, rename your `.tflite` file to
`1_<your model named>.tflite` and copy it into the config/tflite folder of the Simplicity Studio
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
mltk_set(GECKO_SDK_ENABLE_BLUETOOTH ON)
mltk_set(BLE_AUDIO_CLASSIFIER_MODEL <model name or path>)
```

where `<model name or path>` is the file path to your model's `.tflite` 
or the MLTK model name.

With this variable set, when the ble_audio_classifier application is built the 
specified model will be built into the application.


## Build, Run, Debug

See the [online documentation](https://siliconlabs.github.io/mltk/docs/cpp_development/index.html) for how to build and run this application:


### Simplicity Studio

If using [Simplicity Studio](https://siliconlabs.github.io/mltk/docs/cpp_development/simplicity_studio.html) select the `MLTK - BLE Audio Classifier` Project.

### Visual Studio Code


If using [Visual Studio Code](https://siliconlabs.github.io/mltk/docs/cpp_development/vscode.html) select the `mltk_ble_audio_classifier` CMake target.

To build this app, create/modify the file: `<mltk repo root>/user_options.cmake`
and add:

```
mltk_set(GECKO_SDK_ENABLE_BLUETOOTH ON)
```


### Command-line

If using the [Command Line](https://siliconlabs.github.io/mltk/docs/cpp_development/command_line.html) select the `mltk_ble_audio_classifier` CMake target.  

To build this app, create/modify the file: `<mltk repo root>/user_options.cmake`
and add:

```
mltk_set(GECKO_SDK_ENABLE_BLUETOOTH ON)
```


## Model Parameters

In order for the audio classification to work correctly, we need to use the same
audio feature generator configuration parameters for inference as is used when
training the model. When using the MLTK to train an audio classification model
the model [parameters](https://siliconlabs.github.io/mltk/docs/guides/model_parameters.html#audiodatasetmixin) 
will be embedded in the metadata section of the `.tflite`
file. The model parameters are extracted from the `.tflite` at runtime.


## Additional Reading

- [MLTK Documentation](https://siliconlabs.github.io/mltk)
- [QSG169: Bluetooth SDK v3.x Quick Start Guide](https://www.silabs.com/documents/public/quick-start-guides/qsg169-bluetooth-sdk-v3x-quick-start-guide.pdf)
- [Bluetooth Documentation](https://docs.silabs.com/bluetooth/latest/)
- [UG103.14: Bluetooth LE Fundamentals](https://www.silabs.com/documents/public/user-guides/ug103-14-fundamentals-ble.pdf)
- [UG434: Silicon Labs Bluetooth ® C Application Developer's Guide for SDK v3.x](https://www.silabs.com/documents/public/user-guides/ug434-bluetooth-c-soc-dev-guide-sdk-v3x.pdf)
- [Bluetooth Training](https://www.silabs.com/support/training/bluetooth)
- [Audio Feature Generator](https://siliconlabs.github.io/mltk/docs/audio/audio_feature_generator.html)
- [Audio Utilities](https://siliconlabs.github.io/mltk/docs/audio/audio_utilities.html)
- [Gecko SDK Machine Learning Documentation](https://docs.silabs.com/gecko-platform/latest/machine-learning/tensorflow/getting-started)
- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)
