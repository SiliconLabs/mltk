# Tensorflow-Lite Micro Library

This is the [Tensorflow-Lite Micro](https://github.com/tensorflow/tflite-micro) (TFLM) library plus additional helper functions.


__NOTE:__ The actual Tensorflow-Lite Micro [library](https://github.com/tensorflow/tflite-micro) is downloaded by the CMake [build script](./CMakeLists.txt).


## Version

The current TFLM version used by the MLTK may be found in [CMakeLists.txt](./CMakeLists.txt), in the function:

```
CPMAddPackage(
NAME Tensorflow
  GITHUB_REPOSITORY tensorflow/tflite-micro
  GIT_TAG <commit>
  DOWNLOAD_ONLY ON
  CACHE_SUBDIR tflite_micro
  CACHE_VERSION ${_tflm_cache_version}
)
```

where `GIT_TAG <commit>` points to the GIT commit that is downloaded.


## Modifications

The following files were overridden:  
- [micro_graph.cc](./micro_graph.cc)
- [simple_memory_allocator.cc](./simple_memory_allocator.cc)

The [patch_tensorflow.py](./patch_tensorflow.py) Python script is also used to make modifications to the TFLM source code.


## Additional Links

- [C++ Development documentation](https://siliconlabs.github.io/mltk/docs/cpp_development/index.html)
- [Python wrapper documentation](https://siliconlabs.github.io/mltk/docs/cpp_development/wrappers/tflite_micro_wrapper.html)
