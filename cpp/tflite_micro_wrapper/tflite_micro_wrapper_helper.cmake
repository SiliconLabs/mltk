
# This is the name of the generate DLL/shared library
# This is always the name of the wrapper's Python module
# e.g.: import _tflite_micro_wrapper
set(TFLITE_MICRO_MODULE_NAME _tflite_micro_wrapper CACHE INTERNAL "")


# Set the TF-Lite Micro API version
# This is used to ensure accelerator wrappers
# are compatbile with the TFLM wrapper.
# This should change if the TF-Lite Micro MicroOpResolver
# API has breaking changes
set(TFLITE_MICRO_API_VERSION 1 CACHE INTERNAL "")

# Enable MLTK profiling support in TFLM
# NOTE: This must be set BEFORE the tflm package is included by CMake below
mltk_set(TFLITE_MICRO_PROFILER_ENABLED ON)
# Enable MLTK tensor recording support in TFLM
# NOTE: This must be set BEFORE the tflm package is included by CMake below
mltk_set(TFLITE_MICRO_RECORDER_ENABLED ON)
# Enable the hardware simulator
# NOTE: This must be set BEFORE the tflm package is included by CMake below
mltk_set(TFLITE_MICRO_SIMULATOR_ENABLED ON)


set(TFLITE_MICRO_WRAPPER_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}" CACHE INTERNAL "")
set(TFLITE_MICRO_WRAPPER_FULLNAME "${PYTHON_MODULE_PREFIX}${TFLITE_MICRO_MODULE_NAME}${PYTHON_MODULE_EXTENSION}" CACHE INTERNAL "")
set(TFLITE_MICRO_WRAPPER_DIR "${MLTK_DIR}/core/tflite_micro" CACHE INTERNAL "")
set(TFLITE_MICRO_WRAPPER_ACCELERATORS_DIR "${TFLITE_MICRO_WRAPPER_DIR}/accelerators" CACHE INTERNAL "")
set(TFLITE_MICRO_WRAPPER_PATH "${TFLITE_MICRO_WRAPPER_DIR}/${TFLITE_MICRO_WRAPPER_FULLNAME}" CACHE INTERNAL "")
set(TFLITE_MICRO_WRAPPER_IMPORT_LIB_PATH "${MLTK_BINARY_DIR}/${TFLITE_MICRO_WRAPPER_FULLNAME}.a" CACHE INTERNAL "")


# Strip all symbols from built objects
# This makes the built .pyd/.so smaller but non-debuggable
# Comment this line if you want to enable debugging of the shared library
mltk_get(TFLITE_MICRO_WRAPPER_ENABLE_DEBUG_SYMBOLS)
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug" OR TFLITE_MICRO_WRAPPER_ENABLE_DEBUG_SYMBOLS)
  set(TFLITE_MICRO_WRAPPER_ENABLE_OPTIMIZATION OFF)
  mltk_warn("Building wrapper with debug symbols")
else()
  mltk_append_global_cxx_flags("-s")
  mltk_info("Stripping debug symbols from wrapper")
  set(TFLITE_MICRO_WRAPPER_ENABLE_OPTIMIZATION ON)
endif()

# Add OS-specific build flags
if(HOST_OS_IS_WINDOWS)
  set(additional_libs ws2_32 wsock32 )
  mltk_append_global_cxx_flags("-fvisibility=hidden")
  mltk_append_global_cxx_defines("MLTK_DLL_EXPORT")
else()
  # Ensure all source files are built with the Position-Independent-Code (PIC) flag
  mltk_append_global_cxx_flags("-fPIC")
endif()


# Return the current GIT hash
# This will be embedded into the generated wrapper library
mltk_git_hash(${CMAKE_CURRENT_LIST_DIR} MLTK_GIT_HASH)
mltk_info("Git hash: ${MLTK_GIT_HASH}")


###########################################################################################
# tflite_micro_link_python_wrapper
# 
# Link the TF-Lite Micro python wrapper to another shared library
# This must be invoked by accelerator Python wrappers
#
# target - Shared library CMake build target
# lib_dir - Directory where shared library will reside in MLTK
#           This is needed for Linux to enable relative imports
function(tflite_micro_link_python_wrapper target lib_dir)
  # Add this global define so that tflite_micro_wrapper DLL APIs are used
  mltk_append_global_cxx_defines("MLTK_DLL_IMPORT")
  
  if(HOST_OS_IS_WINDOWS)
    # Windows requires an "import library" to properly link against the tflite_micro_wrapper DLL
    # So first generate a .def from the .dll (which has a .pyd extension)
    # then, generate a .a from the .def and .pyd
    string(REGEX REPLACE "\.pyd$" "" fullname_no_ext ${TFLITE_MICRO_WRAPPER_FULLNAME})
   
    # FIXME: If the generated .a does not go in the correct directory,
    # then the generated .pyd may not import the shared symbols properly
    # at runtime. This seems to be dependent on the build directory but
    # it is not obvious what the path should to properly link...
    #if("${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}" STREQUAL "37")
    #  get_filename_component(mltk_build_dir "${MLTK_DIR}/../build" ABSOLUTE)
    #  file(MAKE_DIRECTORY ${mltk_build_dir})
    #  set(tflite_micro_archive_path "${mltk_build_dir}/${fullname_no_ext}.a")
    #else()
    set(tflite_micro_archive_path "${CMAKE_CURRENT_BINARY_DIR}/${fullname_no_ext}.a")
    #endif()
    mltk_info("tflite_micro_archive_path: ${tflite_micro_archive_path}")
    mltk_info("${CMAKE_GENDEF} ${TFLITE_MICRO_WRAPPER_PATH}")
    mltk_info("${CMAKE_DLLTOOL} -k -d ${CMAKE_CURRENT_BINARY_DIR}/${fullname_no_ext}.def -l ${tflite_micro_archive_path}")
    add_custom_command( 
      COMMAND ${CMAKE_GENDEF} ${TFLITE_MICRO_WRAPPER_PATH}
      COMMAND ${CMAKE_DLLTOOL} -k -d ${CMAKE_CURRENT_BINARY_DIR}/${fullname_no_ext}.def -l ${tflite_micro_archive_path}
      OUTPUT ${tflite_micro_archive_path}
      COMMENT "Generating ${tflite_micro_archive_path}"
    )
    add_custom_target(
      tflite_micro_python_wrapper_shared_generate ALL 
      DEPENDS ${tflite_micro_archive_path}
    )
    add_library(tflite_micro_python_wrapper_shared STATIC IMPORTED)
    set_target_properties(tflite_micro_python_wrapper_shared
    PROPERTIES 
      IMPORTED_LOCATION "${tflite_micro_archive_path}"
    )
    add_dependencies(tflite_micro_python_wrapper_shared tflite_micro_python_wrapper_shared_generate)
    
  else()
    add_library(tflite_micro_python_wrapper_shared INTERFACE)
    # The following allows for an accelerator wrapper shared libraries to find symbols
    # in the tflite_micro_wrapper shared library
    file(RELATIVE_PATH module_relpath ${lib_dir} ${TFLITE_MICRO_WRAPPER_PATH})
    get_filename_component(module_reldir ${module_relpath} DIRECTORY)
    mltk_info("tflite_micro_wrapper relative directory: ${module_reldir}")

    target_link_libraries(tflite_micro_python_wrapper_shared
    INTERFACE
      "-Wl,-z,origin"
      "-Wl,-rpath=$$ORIGIN/${module_reldir}"
      "-Wl,-L${TFLITE_MICRO_WRAPPER_DIR}"
      "-Wl,-l:${TFLITE_MICRO_WRAPPER_FULLNAME}"
    )
  endif()

  target_link_libraries(${target}
  PUBLIC 
    tflite_micro_python_wrapper_shared
  )

  target_include_directories(${target}
  PRIVATE 
    ${TFLITE_MICRO_WRAPPER_SOURCE_DIR}
  )

endfunction()