# Fingerprint Authenticator

This application uses machine learning to generate a unique "signature" from a grayscale image of a person's fingerprint captured by the [R503 Fingerprint Module](https://www.adafruit.com/product/4651). The generated signature is then compared against previously generated signatures saved to flash memory. If a saved signature is similar to the generated signature then the user is considered authenticated.


__NOTE:__ This application _only_ supports running on supported embedded targets (Windows/Linux is not supported).


## Quick Links

- [GitHub Source](https://github.com/SiliconLabs/mltk/tree/master/cpp/shared/apps/fingerprint_authenticator) - View this example's source code on Github
- [Hardware Setup](#hardware-setup) - View the required hardware setup for this example
- [Online documentation](https://siliconlabs.github.io/mltk/docs/cpp_development/examples/fingerprint_authenticator.html) - View this example's online documentation
- [Fingerprint Authentication Tutorial](https://siliconlabs.github.io/mltk/mltk/tutorials/fingerprint_authentication.html) - View this example's associated tutorial
- [R503 Fingerprint Module](https://www.adafruit.com/product/4651) - View the fingerprint reader's product page


## Video

A demo video of this application may be found here:  
[![Fingerprint Authenticator Video](https://img.youtube.com/vi/RD_K1UA_Sp8/0.jpg)](https://www.youtube.com/watch?v=RD_K1UA_Sp8)



## Behavior

This application has the following behavior:

* Click button 2 to iterate through users  
  The LED will be solid red, blue, or purple to signify the current user:
  - RED    -> user 0
  - BLUE   -> user 1
  - PURPLE -> user 2

* Press button 2 for 10s then release to erase the current user's signatures.  
  The LED will pulse purple while the erase sequence initializes,
  and flash purple when the signatures are erased.  
  Release button 2 before 10s have elapsed to abort the sequence.

* Click button 1 to save the fingerprints for the current user.  
  The LED will flash blue when you should place your finger on the reader.  
  If the LED flashes red then there was a reading error,
  wait for the LED to flash blue to try again.  
  This sequence will repeat 3 times.
  i.e. The SAME finger will be captured 3 times.  
  If there is no activity on the reader after 7s,
  then this sequence will be aborted.

* In normal operation, the LED pulses blue.  
  Place your finger on the reader to authenticate.  
  The LED will pulse purple while your finger is processed.  
  The LED will be solid red, blue, or purple for the authenticated user.  
  The LED will flash purple for an unknown fingerprint


__HINT:__ Run the following command to view the captured fingerprints:

```shell
mltk fingerprint_reader
```

### State Diagram

A state diagram of this application is as follows:

![](https://siliconlabs.github.io/mltk/img/fingerprint_authenticator_state_diagram.png)


## Signature Generation Model

The application is designed to run with a model similar to [fingerprint_signature_generator](https://siliconlabs.github.io/mltk/docs/python_api/models/siliconlabs/fingerprint_signature_generator.html)

This model receives a pre-processed grayscale image of the fingerprint and generates its
corresponding unique signature.

Refer to [Fingerprint Authentication Tutorial](https://siliconlabs.github.io/mltk/mltk/tutorials/fingerprint_authentication.html) for more details on how to create this model.



## Updating the model

The application will run _any_ quantized `.tflite` model file. 
A default model comes with the application, however, this model may be updated 
using several different methods:


### via Simplicity Studio

To replace the default model, rename your `.tflite` file to
`1_<your model named>.tflite` and copy it into the config/tflite folder of Simplicity Studio
project. (Simplicity Studio sorts the models alphabetically in ascending order, adding `1_` 
forces the model to come first). After a new .tflite file is added to the 
project Simplicity Studio will automatically use the 
[flatbuffer converter tool](https://docs.silabs.com/gecko-platform/latest/machine-learning/tensorflow/flatbuffer-conversion)
to convert a .tflite file into a c file which is added to the project.

Refer to the online [documentation](https://docs.silabs.com/gecko-platform/latest/machine-learning/tensorflow/guide-replace-model#updating-or-replacing-the--tflite-file-in-a-project) for more details.


#### via fingeprint_reader Command

Alternatively, using the command:

```shell
mltk fingeprint_reader <model path> --app none
```

will program the `.tflite` model to the end of the device's flash.
On startup, the application will detect the new model and use that instead
of the model built into the firmware.

__NOTE:__ The `--app none` option tells the command to _not_ update the fingeprint_authenticator application and only program the model file.


### via CMake

The model can also be updated when building this application from [Visual Studio Code](https://siliconlabs.github.io/mltk/docs/cpp_development/vscode.html)
or the CMake [Command Line](https://siliconlabs.github.io/mltk/docs/command_line/index.html).

To update the model, create/modify the file: `<mltk repo root>/user_options.cmake`
and add:

```shell
mltk_set(FINGERPRINT_AUTHENTICATOR_MODEL <model name or path>)
```

where `<model name or path>` is the file path to your model's `.tflite` 
or the MLTK model name.

With this variable set, when the fingeprint_authenticator application is built the 
specified model will be built into the application.



## Hardware Setup

To run this application, an [R503 Fingerprint Module](https://www.adafruit.com/product/4651) is required.
This module must be connected to the USART peripheral of the embedded target.

The following default pin mappings are used by:  
-  __BRD2601__

| R503 Pin              | Board Expansion Header Pin |
| --------------------- | -------------------------- |
| GND          (black)  | 1                          |
| Power Supply (red)    | 20                         |
| 3.3VT        (white)  | 20                         |
| TXD          (yellow) | 6                          |
| RXD          (green)  | 4                          |
| Wakeup       (blue)   | 10                         |



## Build, Run, Debug

See the [online documentation](https://siliconlabs.github.io/mltk/docs/cpp_development/index.html) for how to build and run this application:


### Simplicity Studio

If using [Simplicity Studio](https://siliconlabs.github.io/mltk/docs/cpp_development/simplicity_studio.html) select the `MLTK - Fingerprint Authenticator` Project.

### Visual Studio Code
If using [Visual Studio Code](https://siliconlabs.github.io/mltk/docs/cpp_development/vscode.html) select the `mltk_fingerprint_authenticator` CMake target.

### Command-line

If using the [Command Line](https://siliconlabs.github.io/mltk/docs/cpp_development/command_line.html) select the `mltk_fingerprint_authenticator` CMake target.  



## Dumping images to PC

This application works with the MLTK command:

```shell
mltk fingerprint_reader --help
```

Using this command, you can dump images to the local PC.

For example:

```shell
mltk classify_image fingerprint_reader --dump-images
```

which will display the images from the fingerprint reader and save them to the local PC



## Build Settings

When building this application using [Visual Studio Code](https://siliconlabs.github.io/mltk/docs/cpp_development/vscode.html) 
or the [Command Line](https://siliconlabs.github.io/mltk/docs/cpp_development/command_line.html) several options may be configured 
to control how the app is built.

To specify the settings, create/modify the file:  
`<mltk repo root>/user_options.cmake`



The following settings are supported:


### FINGERPRINT_AUTHENTICATOR_MODEL

Optionally, configure the `.tflite` model to profile:

```shell
# Specify the path to the .tflite model file to use
# by the mltk_fingerprint_authenticator application
mltk_set(FINGERPRINT_AUTHENTICATOR_MODEL ~/my_models/my_model.tflite)
```

__HINT:__  
You can also specify the path to the `.mltk.zip` model archive or just specify the MLTK model name, e.g.:

```shell
# Specify the path to the model archive
mltk_set(FINGERPRINT_AUTHENTICATOR_MODEL ~/my_models/my_model.mltk.zip)

# Specify the MLTK model name
# NOTE: The model specification must be on the model search path, see:
#       https://siliconlabs.github.io/mltk/docs/guides/model_search_path.html
mltk_set(FINGERPRINT_AUTHENTICATOR_MODEL fingerprint_signature_generator)
```

### TFLITE_MICRO_ACCELERATOR

Optionally, configure the target hardware accelerator:

```shell
# Use the Gecko SDK MVP TFLM kernels
mltk_set(TFLITE_MICRO_ACCELERATOR mvp)
```
