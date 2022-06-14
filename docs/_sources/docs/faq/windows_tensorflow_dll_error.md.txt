# Windows: I am seeing a DLL error when importing Tensorflow

When running the MLTK, if you are seeing an error similar to:

> ImportError: DLL load failed while importing _pywrap_tensorflow_internal: The specified module could not be found.

This could be due to several different issues:

## Your setup is missing a required DLL

Ensure your setup has the latest Visual C++ Redistributable binaries.  
You may download the latest from here:  
https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads

[x64 Download Link](https://aka.ms/vs/17/release/vc_redist.x64.exe)


## Your setup requires the NVIDIA CUDA driver

Refer to the Tensorflow [install instructions](https://www.tensorflow.org/install/pip) for how to install the driver.


