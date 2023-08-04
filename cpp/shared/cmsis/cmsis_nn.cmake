

CPMAddPackage(
    NAME cmsis_nn
    URL https://github.com/ARM-software/CMSIS-NN/archive/dc64e488f6655aa2792d2aceca316c896f78b4db.zip
    URL_HASH SHA1=7cc7f8391bc29f584e661ea7599cd01d9569d169
    DOWNLOAD_ONLY ON
    CACHE_SUBDIR cmsis
    CACHE_VERSION may23_2023
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
