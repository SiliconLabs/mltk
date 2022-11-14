project(mltk_audio_feature_generator_wrapper
        VERSION 1.0.0
        DESCRIPTION "MLTK Audio Feature Generator Python wrapper"
)
export(PACKAGE ${PROJECT_NAME})


# Audio Feature Generator API version
# Increment this for any major changes to the C++ wrapper API
# This ensure the Python is compatible wih the C++ wrapper
set(AUDIO_FEATURE_GENERATOR_API_VERSION 1)

####################################################
# This is only support for non-embedded platforms
# So if we're building for embedded then immediately reutrn
mltk_get(MLTK_PLATFORM_IS_EMBEDDED)
if(MLTK_PLATFORM_IS_EMBEDDED)
    return()
endif()

####################################################
# This is the name of the generate DLL/shared library
set(MODULE_NAME _audio_feature_generator_wrapper)
add_custom_target(${PROJECT_NAME}
  DEPENDS ${MODULE_NAME}
)

####################################################
# Return the current GIT hash
# This will be embedded into the generated wrapper library
mltk_git_hash(${CMAKE_CURRENT_LIST_DIR} MLTK_GIT_HASH)
mltk_info("Git hash: ${MLTK_GIT_HASH}")

####################################################
# Find the CMake components required by this wrapper
find_package(mltk_pybind11 REQUIRED)
find_package(mltk_microfrontend REQUIRED)


####################################################
# Define the tflite_micro_wrapper pybind11 wrapper target
pybind11_add_module(${MODULE_NAME} 
  audio_feature_generator_wrapper_pybind11.cc
  audio_feature_generator_wrapper.cc
)

# Strip all symbols from built objects
# This makes the built .pyd/.so smaller but non-debuggable
# Comment this line if you want to enable debugging of the shared library
if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  mltk_append_global_cxx_flags("-s")
endif()


# Add OS-specific build flags
if(HOST_OS_IS_WINDOWS)
  mltk_append_global_cxx_flags("-fvisibility=hidden")
  mltk_append_global_cxx_defines("MLTK_DLL_EXPORT")
  target_link_options(${MODULE_NAME}
  PUBLIC
    -static-libgcc -static-libstdc++ -static
  )
else()
  # Ensure all source files are built with the PIC flag
  mltk_append_global_cxx_flags("-fvisibility=hidden -fPIC")
  # Ensure we statically link to the C/C++ libs
  # to reduce run-time dependencies
  target_link_options(${MODULE_NAME}
  PUBLIC
    -static-libgcc -static-libstdc++
  )
  mltk_platform_linux_link_legacy_glibc()
endif()


# Set additional build properties
set_target_properties(${MODULE_NAME} PROPERTIES
  OPT_SIZE ON
  INTERPROCEDURAL_OPTIMIZATION ON
  CXX_VISIBILITY_PRESET hidden
  VISIBLITY_INLINES_HIDDEN ON
)

# Add #defines to tflite_micro_wrapper_pybind11.cc  
set_property(
  SOURCE audio_feature_generator_wrapper_pybind11.cc 
  PROPERTY COMPILE_DEFINITIONS
  MODULE_NAME=${MODULE_NAME}
  AUDIO_FEATURE_GENERATOR_API_VERSION=${AUDIO_FEATURE_GENERATOR_API_VERSION}
  GIT_HASH="${GIT_HASH}"
)

target_link_libraries(${MODULE_NAME} 
PUBLIC 
  mltk::microfrontend
)

target_include_directories(${MODULE_NAME} 
PUBLIC 
  ${CMAKE_CURRENT_LIST_DIR}
)

target_compile_features(${MODULE_NAME}  
PUBLIC 
  cxx_std_17
)

target_link_options(${MODULE_NAME}
PUBLIC
  -Wl,-Map,${CMAKE_CURRENT_BINARY_DIR}/output.map
)

if(HOST_OS_IS_WINDOWS)
  target_link_options(${MODULE_NAME}
  PUBLIC
    -static-libgcc -static-libstdc++ -pthread -static
  )
endif()


# Copy the built .pyd/.so to the directory:
# <mltk root>/mltk/core/preprocess/audio/audio_feature_generator
set(audio_feature_generator_dir "${MLTK_DIR}/core/preprocess/audio/audio_feature_generator")
add_custom_command(
  TARGET ${MODULE_NAME} 
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${MODULE_NAME}> ${audio_feature_generator_dir}
  COMMAND ${CMAKE_COMMAND} -E echo "Copying built wrapper to ${audio_feature_generator_dir}/$<TARGET_FILE_NAME:${MODULE_NAME}>"
)
