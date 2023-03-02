add_library(mltk_cmsis_nn INTERFACE)
add_library(mltk::cmsis_nn ALIAS mltk_cmsis_nn)


set(CMSIS_PATH ${cmsis_SOURCE_DIR})
add_subdirectory(${cmsis_SOURCE_DIR}/CMSIS/NN/Source)


target_link_libraries(mltk_cmsis_nn
INTERFACE
  cmsis-nn
  mltk::cmsis_dsp
)

target_include_directories(mltk_cmsis_nn
INTERFACE
  ${cmsis_SOURCE_DIR}/CMSIS/NN
)

target_compile_definitions(mltk_cmsis_nn
INTERFACE
  CMSIS_NN=1
)