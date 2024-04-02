

CPMAddPackage(
    NAME cmsis_nn
    URL https://github.com/ARM-software/CMSIS-NN/archive/refs/tags/24.02.zip
    URL_HASH SHA1=42cffeae7e05cb07de844774cd92ff6ec2c0e8dc
    DOWNLOAD_ONLY ON
    CACHE_SUBDIR cmsis
    CACHE_VERSION feb11_2024
)
set(cmsis_nn_SOURCE_DIR ${cmsis_nn_SOURCE_DIR} CACHE INTERNAL "")


if(MLTK_PLATFORM_IS_EMBEDDED)
  add_subdirectory(${cmsis_nn_SOURCE_DIR})

  add_library(mltk_cmsis_nn INTERFACE)
  add_library(mltk::cmsis_nn ALIAS mltk_cmsis_nn)


  target_link_libraries(mltk_cmsis_nn
  INTERFACE
    mltk::cmsis_dsp
  )

  target_include_directories(mltk_cmsis_nn
  INTERFACE
    ${cmsis_nn_SOURCE_DIR}
  )


  mltk_append_global_cxx_defines(CMSIS_NN_USE_SINGLE_ROUNDING)
  target_compile_definitions(mltk_cmsis_nn
  INTERFACE
    CMSIS_NN=1
  )

  target_link_libraries(mltk_cmsis_nn
  INTERFACE
    cmsis-nn
  )

endif() # MLTK_PLATFORM_IS_EMBEDDED
