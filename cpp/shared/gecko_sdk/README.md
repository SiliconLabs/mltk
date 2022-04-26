# Gecko SDK

This allows for building the [Gecko SDK](https://github.com/SiliconLabs/gecko_sdk/tree/gsdk_4.0) (GSDK) into applications.

__NOTE:__ The actual Gecko SDK source code is downloaded by the CMake [build script](./CMakeLists.txt).



## Version

The current GSDK version used by the MLTK may be found in [CMakeLists.txt](./CMakeLists.txt), in the function:

```
CPMAddPackage(
  NAME gecko_sdk
  GITHUB_REPOSITORY SiliconLabs/gecko_sdk
  GIT_TAG <commit>
  CACHE_VERSION v4.0.2
  DOWNLOAD_ONLY ON
)
```

where `GIT_TAG <commit>` points to the GIT commit that is downloaded.



## Additional Links

- [C++ Development documentation](https://siliconlabs.github.io/mltk/docs/cpp_development/index.html)
