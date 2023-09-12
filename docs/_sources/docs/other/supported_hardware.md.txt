# Supported Hardware

Various MLTK commands such as [profile](../guides/model_profiler.md) and [classify_audio](../audio/audio_utilities.md) support executing on the desktop (e.g. Windows/Linux)
_or_ on supported embedded platforms. Additionally, [C++ applications](../cpp_development/index.md) may be developed for the desktop
as well as supported embedded platforms.

The following embedded platforms are currently supported:


## BRD2601

- Name: EFR32xG24 Dev Kit
- [Product Link](https://www.silabs.com/development-tools/wireless/efr32xg24-dev-kit)


To build a C++ application for this platform using [VSCode](../cpp_development/vscode.md) or [Command-Line](../cpp_development/command_line.md), create/modify `<mltk repo root>/user_options.cmake`:

```
mltk_set(MLTK_PLATFORM_NAME brd2601)
```

This platform features the [MVP](https://docs.silabs.com/gecko-platform/latest/machine-learning/tensorflow/mvp-accelerator) machine learning hardware accelerator.

To build a C++ application with MVP hardware acceleration using [VSCode](../cpp_development/vscode.md) or [Command-Line](../cpp_development/command_line.md), create/modify `<mltk repo root>/user_options.cmake`:

```
mltk_set(TFLITE_MICRO_ACCELERATOR mvp)
```

The platform also supports the following [commands](../command_line/index.md) when using `--device` command line option:

- [profile](../guides/model_profiler.md)
- [classify_audio](../audio/audio_utilities.md)
- [classify_image](../../mltk/tutorials/image_classification)
- [fingerprint_reader](../../mltk/tutorials/fingerprint_authentication)


## BRD2204

- Name: EFM32 Giant Gecko S1, GG11 Starter Kit
- [Product Link](https://www.silabs.com/development-tools/mcu/32-bit/efm32gg11-starter-kit)


To build a C++ application for this platform using [VSCode](../cpp_development/vscode.md) or [Command-Line](../cpp_development/command_line.md), create/modify `<mltk repo root>/user_options.cmake`:

```
mltk_set(MLTK_PLATFORM_NAME brd2204)
```

The platform also supports the following [commands](../command_line/index.md) when using `--device` command line option:

- [profile](../guides/model_profiler.md)
- [classify_audio](../audio/audio_utilities.md)


## BRD4166

- Name: Thunderboard Sense 2
- [Product Link](https://www.silabs.com/development-tools/thunderboard/thunderboard-sense-two-kit)



To build a C++ application for this platform using [VSCode](../cpp_development/vscode.md) or [Command-Line](../cpp_development/command_line.md), create/modify `<mltk repo root>/user_options.cmake`:

```
mltk_set(MLTK_PLATFORM_NAME brd4166)
```

The platform also supports the following [commands](../command_line/index.md) when using `--device` command line option:

- [profile](../guides/model_profiler.md)
- [classify_audio](../audio/audio_utilities.md)


## BRD4186

- Name: EFR32xG24 Pro Kit +10 dBm
- [Product Link](https://www.silabs.com/wireless/zigbee/efr32mg24-series-2-socs/device.efr32mg24b210f1536im48)


To build a C++ application for this platform using [VSCode](../cpp_development/vscode.md) or [Command-Line](../cpp_development/command_line.md), create/modify `<mltk repo root>/user_options.cmake`:

```
mltk_set(MLTK_PLATFORM_NAME brd4186)
```

This platform features the [MVP](https://docs.silabs.com/gecko-platform/latest/machine-learning/tensorflow/mvp-accelerator) machine learning hardware accelerator.

To build a C++ application with MVP hardware acceleration using [VSCode](../cpp_development/vscode.md) or [Command-Line](../cpp_development/command_line.md), create/modify `<mltk repo root>/user_options.cmake`:

```
mltk_set(TFLITE_MICRO_ACCELERATOR mvp)
```

The platform also supports the following [commands](../command_line/index.md) when using `--device` command line option:

- [profile](../guides/model_profiler.md)


## BRD4401

- Name: EFR32xG28 2.4 GHz BLE and +20 dBm Radio Board
- [Product Link](https://www.silabs.com/development-tools/wireless/xg28-rb4401c-efr32xg28-2-4-ghz-ble-and-20-dbm-radio-board)


To build a C++ application for this platform using [VSCode](../cpp_development/vscode.md) or [Command-Line](../cpp_development/command_line.md), create/modify `<mltk repo root>/user_options.cmake`:

```
mltk_set(MLTK_PLATFORM_NAME brd4401)
```

This platform features the [MVP](https://docs.silabs.com/gecko-platform/latest/machine-learning/tensorflow/mvp-accelerator) machine learning hardware accelerator.

To build a C++ application with MVP hardware acceleration using [VSCode](../cpp_development/vscode.md) or [Command-Line](../cpp_development/command_line.md), create/modify `<mltk repo root>/user_options.cmake`:

```
mltk_set(TFLITE_MICRO_ACCELERATOR mvp)
```

The platform also supports the following [commands](../command_line/index.md) when using `--device` command line option:

- [profile](../guides/model_profiler.md)
