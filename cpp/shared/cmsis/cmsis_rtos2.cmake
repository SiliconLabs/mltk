
add_library(mltk_cmsis_rtos2)
add_library(mltk::cmsis_rtos2 ALIAS mltk_cmsis_rtos2)

target_sources(mltk_cmsis_rtos2
PRIVATE
    ${cmsis_SOURCE_DIR}/CMSIS/RTOS2/Source/os_systick.c
)
target_include_directories(mltk_cmsis_rtos2
PUBLIC
    ${cmsis_SOURCE_DIR}/CMSIS/RTOS2/Include
)
target_link_libraries(mltk_cmsis_rtos2
PRIVATE
    ${MLTK_PLATFORM}
)