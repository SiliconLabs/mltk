

include(${CMAKE_CURRENT_LIST_DIR}/../cmsis_dsp.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../cmsis_nn.cmake)

add_library(mltk_cmsis_nn)
add_library(mltk::cmsis_nn ALIAS mltk_cmsis_nn)

target_include_directories(mltk_cmsis_nn
PUBLIC
    ${cmsis_SOURCE_DIR}
    ${cmsis_SOURCE_DIR}/CMSIS/Core/Include
    ${cmsis_nn_SOURCE_DIR}/Include
    ${cmsis_nn_SOURCE_DIR}
    ${cmsis_dsp_SOURCE_DIR}/Include
)
target_compile_options(mltk_cmsis_nn
PRIVATE
    -Wno-strict-aliasing
)

target_sources(mltk_cmsis_nn
PRIVATE
    wrapper/get_buffer_size.c
)
target_compile_options(mltk_cmsis_nn
PRIVATE
    -include stdint.h
)
