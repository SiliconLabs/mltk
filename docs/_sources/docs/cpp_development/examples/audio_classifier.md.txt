# Audio Classifier

This application uses TensorFlow Lite for Microcontrollers to run audio
classification machine learning models to classify words from audio data
recorded from a microphone. The detection is visualized using the LED's on the
board and the classification results are written to the VCOM serial port.


__NOTES:__   
- This application is able to be built for Windows/Linux _or_ a supported embedded target.
- This application is designed to be used with the MLTK command:
[mltk classify_audio](https://siliconlabs.github.io/mltk/docs/audio/audio_utilities.html#audio-classification-utility)


## Quick Links

- [GitHub Source](https://github.com/SiliconLabs/mltk/tree/master/cpp/shared/apps/audio_classifier) - View this example's source code on Github
- [Online documentation](https://siliconlabs.github.io/mltk/docs/cpp_development/examples/audio_classifier.html) - View this example's online documentation
- [Keyword Spotting Tutorial](https://siliconlabs.github.io/mltk/mltk/tutorials/keyword_spotting_on_off.html) - View this example's associated tutorial
- [Audio classifier utility](https://siliconlabs.github.io/mltk/docs/audio/audio_utilities.html#audio-classification-utility) - View the audio classifier utility's documentation
- [Alexa Demo](https://siliconlabs.github.io/mltk/mltk/tutorials/keyword_spotting_alexa.html) - Alexa Voice Services (AVS) demo using a Python script


## Behavior

The application is using two LEDs to show detection and activity and it is
printing detection results and debug log output on the VCOM serial port. In the
application configuration file called audio_classifier_config.h the user can
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

The default model used in this application is called `keyword_spotting_on_off_v3.tflite`
and is able to classify audio into 3 different classes labeled "on", "off", "_unknown_". The source for the model can be found here: 
[https://github.com/siliconlabs/mltk/blob/master/mltk/models/siliconlabs/keyword_spotting_on_off_v3.py](https://github.com/siliconlabs/mltk/blob/master/mltk/models/siliconlabs/keyword_spotting_on_off_v3.py)


The application is designed to work with an audio classification model created
using the Silicon Labs Machine Learning Toolkit
([MLTK](https://siliconlabs.github.io/mltk/mltk/tutorials/keyword_spotting_on_off.html)). 
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


### via classify_audio Command

Alternatively, using the `mltk classify_audio <model path> --app none --device`
command program the `.tflite` model to the end of the device's flash.
On startup, the application will detect the new model and use that instead
of the model built into the firmware.

__NOTE:__ The `--app none` option tells the command to _not_ update the audio_classifier application
and only program the model file.

See the [Audio classifier utility](https://siliconlabs.github.io/mltk/docs/audio/audio_utilities.html#audio-classification-utility)
documentation for more details.


### via CMake

The model can also be updated when building this application from [Visual Studio Code](https://siliconlabs.github.io/mltk/docs/cpp_development/vscode.html)
or the CMake [Command Line](https://siliconlabs.github.io/mltk/docs/command_line/index.html).

To update the model, create/modify the file: `<mltk repo root>/user_options.cmake`
and add:

```
mltk_set(AUDIO_CLASSIFIER_MODEL <model name or path>)
```

where `<model name or path>` is the file path to your model's `.tflite` 
or the MLTK model name.

With this variable set, when the audio_classifier application is built the 
specified model will be built into the application.


## Build, Run, Debug

See the [online documentation](https://siliconlabs.github.io/mltk/docs/cpp_development/index.html) for how to build and run this application:


### Simplicity Studio

If using [Simplicity Studio](https://siliconlabs.github.io/mltk/docs/cpp_development/simplicity_studio.html) select the `MLTK - Audio Classifier` Project.

### Visual Studio Code
If using [Visual Studio Code](https://siliconlabs.github.io/mltk/docs/cpp_development/vscode.html) select the `mltk_audio_classifier` CMake target.

### Command-line

If using the [Command Line](https://siliconlabs.github.io/mltk/docs/cpp_development/command_line.html) select the `mltk_audio_classifier` CMake target.  


## Dumping audio & spectrograms to PC

This application works with the MLTK command:

```shell
mltk classify_audio --help
```

Using this command, you can dump spectrograms and recorded audio to the local PC.

For example:

```shell
# Dump spectrograms generated on the embedded device to the local PC
mltk classify_audio keyword_spotting_on_off_v3 --device --dump-spectrograms
```


```shell
# Dump audio recorded by the embedded device to the local PC
mltk classify_audio keyword_spotting_on_off_v3 --device --dump-audio
```

See the [Audio classifier utility](https://siliconlabs.github.io/mltk/docs/audio/audio_utilities.html#audio-classification-utility)
documentation for more details.



## Model Parameters

In order for the audio classification to work correctly, we need to use the same
audio feature generator configuration parameters for inference as is used when
training the model. When using the MLTK to train an audio classification model
the model [parameters](https://siliconlabs.github.io/mltk/docs/guides/model_parameters.html#audiodatasetmixin) 
will be embedded in the metadata section of the `.tflite`
file. The model parameters are extracted from the `.tflite` at runtime.


## Modifications

The application was originally taken from the 
[Gecko SDK](https://github.com/SiliconLabs/gecko_sdk/tree/gsdk_4.0/app/common/example/audio_classifier).

It has been modified as follows:  
1. Supports running on embedded _as well as_ Windows/Linux
2. All relevant `#defines` have been converted to dynamic variables that are
   populated via command line (for Windows/Linux) or from the `.tflite` model parameters
3. `.tflite` can been dynamically loaded via command line (for Windows/Linux)
   or from a `.tflite` programmed to the end of the embedded target's flash
4. Added support for dumping raw microphone audio and generated spectrograms
   for capture by Python script (using the `mltk classify_audio` command)
5. Updated embedded microphone driver to support dynamic sample lengths


## CMake Variables

This application supports the following __optional__ CMake variables.
The variables may be specified on the [command-line](https://siliconlabs.github.io/mltk/docs/cpp_development/command_line.html) or in the [user_options.cmake](https://siliconlabs.github.io/mltk/docs/cpp_development/build_options.html) file.

### VERBOSE

Enable verbose logging while the application executes

```shell
mltk_set(VERBOSE ON)
```

### WINDOW_MS

Configure the length of the averaging window in milliseconds.
This overrides the [model parameter](https://siliconlabs.github.io/mltk/docs/guides/model_parameters.html#audiodatasetmixin) setting `average_window_duration_ms`.
 
```shell
mltk_set(WINDOW_MS 750)
```

### THRESHOLD

Configure the detection threshold. This is a value from 0-255., 255 being the highest.
This overrides the [model parameter](https://siliconlabs.github.io/mltk/docs/guides/model_parameters.html#audiodatasetmixin) setting `detection_threshold`.
 
```shell
mltk_set(THRESHOLD 185)
```

### SUPPRESSION_MS

The amount of time in milliseconds to wait after a detection to begin listening for keywords again.
This overrides the [model parameter](https://siliconlabs.github.io/mltk/docs/guides/model_parameters.html#audiodatasetmixin) setting `suppression_ms`.
 
```shell
mltk_set(SUPPRESSION_MS 500)
```

### COUNT

The minimum number of inference results to average when calculating the detection value
This overrides the [model parameter](https://siliconlabs.github.io/mltk/docs/guides/model_parameters.html#audiodatasetmixin) setting `minimum_count`.
 
```shell
mltk_set(COUNT 2)
```

### VOLUME_GAIN

The integer multiplier value to apply to each microphone sample.
This overrides the [model parameter](https://siliconlabs.github.io/mltk/docs/guides/model_parameters.html#audiodatasetmixin) setting `volume_gain`.
 
```shell
mltk_set(VOLUME_GAIN 2)
```

### LATENCY_MS

This is the amount of time in milliseconds an audio loop takes.
This overrides the [model parameter](https://siliconlabs.github.io/mltk/docs/guides/model_parameters.html#audiodatasetmixin) setting `latency_ms`.
 
```shell
mltk_set(LATENCY_MS 2)
```

## AUDIO_CLASSIFIER_ENABLE_AUDIO_IO

This enables audio input/output streaming via UART.
This is currently only used in the [Keyword Spotting - Alexa](https://siliconlabs.github.io/mltk/mltk/tutorials/keyword_spotting_alexa.html) demo.

```shell
mltk_set(AUDIO_CLASSIFIER_ENABLE_AUDIO_IO ON)
```

__NOTE:__ When this feature is enabled, log prints are effectively disabled.


## Additional Reading

- [MLTK Documentation](https://siliconlabs.github.io/mltk)
- [Audio Feature Generator](https://siliconlabs.github.io/mltk/docs/audio/audio_feature_generator.html)
- [Audio Utilities](https://siliconlabs.github.io/mltk/docs/audio/audio_utilities.html)
- [Gecko SDK Machine Learning Documentation](https://docs.silabs.com/gecko-platform/latest/machine-learning/tensorflow/getting-started)
- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)
