# Gecko SDK MVP-Accelerated Tensorflow Kernels

This library allows for accelerating several Tensorflow-Lite Micro [kernels](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/kernels) using the MVP hardware accelerator.

This was taken from the [Gecko SDK](https://github.com/SiliconLabs/gecko_sdk/tree/gsdk_4.2):  
- [compute](https://github.com/SiliconLabs/gecko_sdk/tree/gsdk_4.2/platform/compute) - MVP peripheral driver and library
- [kernels](https://github.com/SiliconLabs/gecko_sdk/tree/gsdk_4.2/util/third_party/tensorflow_extra/siliconlabs) - Tensorflow-Lite Micro kernels
- simulator - Allows for downloading the pre-built MVPv1 hardware simulator static library

This was slightly modified so that it can be compiled for Windows/Linux.
