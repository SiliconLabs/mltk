####################################################
# This is only support for non-embedded platforms
# So if we're building for embedded then immediately reutrn
mltk_get(MLTK_PLATFORM_IS_EMBEDDED)
if(MLTK_PLATFORM_IS_EMBEDDED)
    return()
endif()


project(mltk_tflite_micro_wrapper
        VERSION 1.0.0
        DESCRIPTION "MLTK TF-Lite Micro Python wrapper"
)
export(PACKAGE ${PROJECT_NAME})

# Include the pybind11 package BEFORE including the helper script
# as it defines PYTHON variables used by the script
find_package(mltk_pybind11 REQUIRED)
# Include the wrapper helper script
include(${CMAKE_CURRENT_LIST_DIR}/tflite_micro_wrapper_helper.cmake)


# Ensure no accelerator is specified
# (accelerators are dynamically loaded from a separate wrapper lib)
mltk_set(TFLITE_MICRO_ACCELERATOR OFF)

# Find the CMake components required by this wrapper
find_package(mltk_tflite_micro_model REQUIRED)


####################################################
# Define the tflite_micro_wrapper pybind11 wrapper target
add_custom_target(${PROJECT_NAME}
  DEPENDS ${TFLITE_MICRO_MODULE_NAME}
)

pybind11_add_module(${TFLITE_MICRO_MODULE_NAME} 
  tflite_micro_wrapper_pybind11.cc
  tflite_micro_model_wrapper_pybind11.cc
  tflite_micro_model_wrapper.cc
)

# Set additional build properties
set_target_properties(${TFLITE_MICRO_MODULE_NAME} PROPERTIES
  OPT_SIZE ${TFLITE_MICRO_WRAPPER_ENABLE_OPTIMIZATION}
  INTERPROCEDURAL_OPTIMIZATION ON
  CXX_VISIBILITY_PRESET hidden
  VISIBLITY_INLINES_HIDDEN ON
)

# Add #defines to tflite_micro_wrapper_pybind11.cc  
set_property(
  SOURCE tflite_micro_wrapper_pybind11.cc 
  PROPERTY COMPILE_DEFINITIONS
  MODULE_NAME=${TFLITE_MICRO_MODULE_NAME}
  TFLITE_MICRO_API_VERSION=${TFLITE_MICRO_API_VERSION}
  MLTK_GIT_HASH="${MLTK_GIT_HASH}"
)


target_include_directories(${TFLITE_MICRO_MODULE_NAME} 
PUBLIC 
  ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(${TFLITE_MICRO_MODULE_NAME} 
PUBLIC 
  ${MLTK_PLATFORM}
  mltk::tflite_micro_model
  mltk::pybind11_helper
)

target_compile_features(${TFLITE_MICRO_MODULE_NAME}  
PUBLIC 
  cxx_std_17
)


target_link_options(${TFLITE_MICRO_MODULE_NAME}
PUBLIC
  -Wl,-Map,${CMAKE_CURRENT_BINARY_DIR}/output.map
)

# If this is a Windows build, then we also need to generate a DLL "import library"
# which other hardware accelerator python wrappers will link against
if(HOST_OS_IS_WINDOWS)
  target_link_options(${TFLITE_MICRO_MODULE_NAME}
  PUBLIC
    -Wl,--out-implib=${TFLITE_MICRO_WRAPPER_IMPORT_LIB_PATH}
    -Wl,--enable-auto-import
    -static-libgcc -static-libstdc++ -pthread -static
  )
else()
  # Ensure we statically link to the C/C++ libs
  # to reduce run-time dependencies
  target_link_options(${TFLITE_MICRO_MODULE_NAME}
  PUBLIC
    -static-libgcc -static-libstdc++ -pthread
  )
endif()


# Copy the built .pyd/.so to the directory:
# <root>/mltk/core/tflite_micro
add_custom_command(
  TARGET ${TFLITE_MICRO_MODULE_NAME} 
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${TFLITE_MICRO_MODULE_NAME}> ${TFLITE_MICRO_WRAPPER_DIR}
  COMMAND ${CMAKE_COMMAND} -E echo "Copying built wrapper to ${TFLITE_MICRO_WRAPPER_DIR}/$<TARGET_FILE_NAME:${TFLITE_MICRO_MODULE_NAME}>"
)
