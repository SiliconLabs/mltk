# Command-Line Development

The MLTK uses [CMake](https://cmake.org/) to build the various C++ libraries and applications.

The following describes how to setup your local PC to build applications via command-line.


## Install Tools

Before we can build MLTK applications, we need to install a few tools first:

### 0) Setup OS

```{eval-rst}
.. tab-set::

   .. tab-item:: Windows

      If you're on Windows, enable `long file paths`:
      https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry

      Also install the Visual Studio C++ compiler (this is needed to build some of the 3rd party Python packages for Python 3.10):
      https://visualstudio.microsoft.com/visual-cpp-build-tools/

      Be sure to check the **Desktop Development with C++** workload in the installer GUI.

   .. tab-item:: Linux

      If you're on Linux, run the following command to install the necessary packages: 

      .. code-block:: shell

         sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
         sudo add-apt-repository -y ppa:deadsnakes/ppa
         sudo apt update
         sudo apt-get -y install build-essential g++-9 cmake ninja-build gdb p7zip-full git-lfs python3-dev python3-venv libusb-1.0-0 libgl1

```

### 1) Install CMake

CMake is a build system utility used to build the C++ applications.  
Install CMake and ensure it is available on the command `PATH`  
More details here: [https://cmake.org/install](https://cmake.org/install)


### 2) Install 7-Zip

7-Zip is a file archiver with a high compression ratio.  
Several of the assets downloaded by the MLTK are compressed in this format.
More details here: [https://www.7-zip.org/download.html](https://www.7-zip.org/download.html)


### 3) Install Python

Install Python 3.7, 3.8, 3.9, _or_ 3.10 64-bit and ensure it is available on the command `PATH`  
More details here: [https://www.python.org/downloads](https://www.python.org/downloads)


### 4) Install GIT

If necessary, install Git:  
[https://git-scm.com/downloads](https://git-scm.com/downloads)


### 5) Clone the MLTK repository

```shell
git clone https://github.com/siliconlabs/mltk.git
```


## Build Sequence

To build from the command-line, the basic sequence is:


### 1) Configure Build Settings

Various build settings may be _optionally_ specified in the file:
```
<mltk repo root>/user_options.cmake
```

Refer to the [Build Options](./build_options.md) for more details on the available settings.


### 2) Configure CMake Project

```shell
# Navigate to the root of the mltk repo
cd mltk

# Configure the CMake project using the desired toolchain
# Windows toolchain
cmake -DCMAKE_TOOLCHAIN_FILE=./cpp/tools/toolchains/gcc/windows/win64_toolchain.cmake -B./build -G Ninja .

# Linux toolchain
cmake -DCMAKE_TOOLCHAIN_FILE=./cpp/tools/toolchains/gcc/linux/linux_toolchain.cmake -B./build -G Ninja .
```

### 3) Build CMake Project for a specific target

```shell
cmake --build ./build --config Release --target mltk_model_profiler
```

### 4) Run the output executable

The built executable will be in the build directory, e.g.:

```shell
./build/mltk_model_profiler.exe
```


## Example Applications

Refer to the [Examples](./examples/index.md) documentation for more details about the applications that come with the MLTK.

## Python Wrappers

Refer to the [Python wrappers](./wrappers/index.md) documentation for more details about the wrappers that come with the MLTK.