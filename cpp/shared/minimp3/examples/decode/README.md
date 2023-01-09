# Mini MP3 Decode Example

This demonstrates of to read and decode an MP3 file and play on a connected speaker via VDAC.


## Prerequisites

- Supported development board, e.g. BRD2601B
- Analog speaker, e.g. https://www.adafruit.com/product/3885


## Setup Steps


0 ) See the [MLTK C++ Development Docs](https://siliconlabs.github.io/mltk/docs/cpp_development/index.html) for setting up your environment

1 ) Connect a supported development board (e.g. BRD2601B) to your PC

2 ) Create or modify the file:

```
<mltk repo root>/user_options.cmake
```

and add the following:

```
mltk_set(MLTK_TARGET mltk_minimp3_example)
```


3 ) Invoke the CMake target: `mltk_minimp3_example_download`

which will build the firmware application and program it to the development board.


When the application starts, will will read and decode the MP3 file (found in `test_mp3_data.h`)
and play it through the `VDAC` speaker.


__NOTE:__ See the `mp3_to_c_header.py` Python script to update the MP3 file.