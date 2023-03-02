cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(mltk
        VERSION 1.0.0
        DESCRIPTION "Silicon Labs Machine Learning Toolkit"
        LANGUAGES C CXX ASM
)
export (PACKAGE mltk)


# Cache the MLTK binary build directory
set(MLTK_BINARY_DIR ${PROJECT_BINARY_DIR} CACHE INTERNAL "MLTK binary build directory")
# Include the MLTK utilities script
include(${CMAKE_CURRENT_LIST_DIR}/cpp/tools/cmake/utilities.cmake)

# Update the CMake module path
# so that we can find other mltk packages via: find_package()
mltk_update_module_path("${CMAKE_CURRENT_LIST_DIR}/cpp/tools/cmake")

# Check if the mltk is being used internally or externally
if("${CMAKE_SOURCE_DIR}" STREQUAL "${PROJECT_SOURCE_DIR}")
  mltk_info("Building from mltk root repository")

  # Update the CMake output directory so built executables
  # and DLL are store in the root of the build directory
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR} CACHE INTERNAL "Build output directory")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR} CACHE INTERNAL "Build output directory")
  set(CMAKE_FIND_PACKAGE_RESOLVE_SYMLINKS ON CACHE INTERNAL "")
  set(CMAKE_INSTALL_LIBDIR ${PROJECT_BINARY_DIR})

  if(NOT DEFINED CPM_SOURCE_CACHE)
      set(CPM_SOURCE_CACHE "${CMAKE_CURRENT_LIST_DIR}/cpp/shared" CACHE INTERNAL "CPM cache directory")
  endif()

else()  # else IF this is being built from an external CMake project

  if (DEFINED CACHE{CPM_SOURCE_CACHE})
    set(CPM_SOURCE_CACHE "${CMAKE_CURRENT_LIST_DIR}/cpp/shared") # Update the variable locally if it's already defined in cache
  else()
    set(CPM_SOURCE_CACHE "${CMAKE_CURRENT_LIST_DIR}/cpp/shared" CACHE INTERNAL "CPM Cache directory")
  endif()

endif() # IF is master project

if(NOT MLTK_USER_OPTIONS)
  set(MLTK_USER_OPTIONS ${CMAKE_CURRENT_LIST_DIR}/user_options.cmake)
endif()

if(EXISTS "${MLTK_USER_OPTIONS}" AND NOT MLTK_NO_USER_OPTIONS)
  mltk_info("MLTK_USER_OPTIONS=${MLTK_USER_OPTIONS}")
  include("${MLTK_USER_OPTIONS}")
endif()

if(MLTK_ADDITIONAL_OPTIONS)
  mltk_info("MLTK_ADDITIONAL_OPTIONS=${MLTK_ADDITIONAL_OPTIONS}")
  foreach(option_path IN LISTS MLTK_ADDITIONAL_OPTIONS)
    include("${option_path}")
  endforeach()
endif()

# Set the default MLTK Cmake log level
if(NOT DEFINED MLTK_CMAKE_LOG_LEVEL)
  mltk_set(MLTK_CMAKE_LOG_LEVEL info)
  mltk_debug("Default MLTK_CMAKE_LOG_LEVEL=${MLTK_CMAKE_LOG_LEVEL}")
endif()

# Include the Cmake Package Manager script
# NOTE: This must be included AFTER settting the
#        CPM_SOURCE_CACHE variable from above
include(${CMAKE_CURRENT_LIST_DIR}/cpp/tools/cmake/CPM.cmake)

###################################################
# Add the MLTK platform package (if necessary)
mltk_add_platform_package()


###################################################
# If no specific build target was given
# then automatically include all known packages
#
# HINT: To build a Python wrapper:
#
# Create the file <mltk root>/user_options.cmake and add:
# mltk_set(MLTK_TARGET mltk_audio_feature_generator_wrapper)
# mltk_set(MLTK_TARGET mltk_tflite_micro_wrapper)
# or
# mltk_set(MLTK_TARGET mltk_mvp_wrapper)
#
# OR:
# To the CMake build string, add:
# MLTK_TARGET=mltk_audio_feature_generator_wrapper
# MLTK_TARGET=mltk_tflite_micro_wrapper
# or
# MLTK_TARGET=mltk_mvp_wrapper

mltk_get(MLTK_TARGET)
if(NOT ("${MLTK_TARGET}" STREQUAL "" OR "${MLTK_TARGET}" STREQUAL "all"))
  mltk_info("MLTK_TARGET=${MLTK_TARGET}")
  mltk_find_package(${MLTK_TARGET})

else()

  find_package(mltk_hello_world)
  find_package(mltk_model_profiler)
  find_package(mltk_audio_classifier)
  find_package(mltk_image_classifier)
  find_package(mltk_fingerprint_authenticator)
  find_package(mltk_ble_audio_classifier)

endif()

