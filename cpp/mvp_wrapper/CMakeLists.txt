####################################################
# This is only support for non-embedded platforms
# So if we're building for embedded then immediately reutrn
mltk_get(MLTK_PLATFORM_IS_EMBEDDED)
if(MLTK_PLATFORM_IS_EMBEDDED)
    return()
endif()


project(mltk_mvp_wrapper
        VERSION 1.0.0
        DESCRIPTION "MLTK MVP Python wrapper"
)
export(PACKAGE ${PROJECT_NAME})

# Enable the accelerator instruction recorder
mltk_set(TFLITE_MICRO_ACCELERATOR_RECORDER_ENABLED ON)

# This is the name of the generate DLL/shared library
# This is always the name of the wrapper's Python module
# e.g.: import _mvp1_wrapper
set(MODULE_NAME _mvp_wrapper)
add_custom_target(${PROJECT_NAME}
  DEPENDS ${MODULE_NAME}
)

# Include the pybind11 package BEFORE including the helper script
# as it defines PYTHON variables used by the script
find_package(mltk_pybind11 REQUIRED)
# Include the tflite_micro_wrapper helper script
include(${CMAKE_CURRENT_LIST_DIR}/../tflite_micro_wrapper/tflite_micro_wrapper_helper.cmake)

# Find the tflite micro mvp kernels which we'll link against
find_package(mltk_tflite_micro_mvp_kernels REQUIRED)
find_package(mltk_tflite_micro_model REQUIRED)


####################################################
# Define the tflite_micro_wrapper pybind11 wrapper target
pybind11_add_module(${MODULE_NAME} 
  mvp_wrapper_pybind11.cc
)

# Set additional build properties
set_target_properties(${MODULE_NAME} PROPERTIES
  OPT_SIZE ${TFLITE_MICRO_WRAPPER_ENABLE_OPTIMIZATION}
  INTERPROCEDURAL_OPTIMIZATION ON
  CXX_VISIBILITY_PRESET hidden
  VISIBLITY_INLINES_HIDDEN ON
)

# Add #defines to tflite_micro_wrapper_pybind11.cc  
set_property(
  SOURCE mvp_wrapper_pybind11.cc 
  PROPERTY COMPILE_DEFINITIONS
  MODULE_NAME=${MODULE_NAME}
  TFLITE_MICRO_API_VERSION=${TFLITE_MICRO_API_VERSION}
  MLTK_GIT_HASH="${MLTK_GIT_HASH}"
)

target_link_libraries(${MODULE_NAME} 
PUBLIC 
  ${MLTK_PLATFORM}
  mltk::tflite_micro_mvp_kernels
)

# Ensure we statically link to the C/C++ libs
# to reduce run-time dependencies
if(HOST_OS_IS_WINDOWS)
  target_link_options(${MODULE_NAME}
  PUBLIC
    -static-libgcc -static-libstdc++ -pthread -static
  )
else()
  target_link_options(${MODULE_NAME}
  PUBLIC
    -static-libgcc -static-libstdc++ -pthread
  )
endif()

# Link the tflite_micro_wrapper shared library
# This must come AFTER the other libs are added above in target_link_libraries()
set(mvp_accelerator_dir "${TFLITE_MICRO_WRAPPER_ACCELERATORS_DIR}/mvp")
tflite_micro_link_python_wrapper(${MODULE_NAME} ${mvp_accelerator_dir})


target_compile_features(${MODULE_NAME}  
PUBLIC 
  cxx_std_17
)

target_link_options(${MODULE_NAME}
PUBLIC
  -Wl,-Map,${CMAKE_CURRENT_BINARY_DIR}/output.map
)


# Copy the built .pyd/.so to the directory:
# <mltk root>/mltk/core/tflite_micro/accelerators/mvp
add_custom_command(
  TARGET ${MODULE_NAME} 
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${MODULE_NAME}> ${mvp_accelerator_dir}
  COMMAND ${CMAKE_COMMAND} -E echo "Copying built wrapper to ${mvp_accelerator_dir}/$<TARGET_FILE_NAME:${MODULE_NAME}>"
)

