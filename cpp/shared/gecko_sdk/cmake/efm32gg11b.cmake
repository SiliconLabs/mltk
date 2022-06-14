set(NAME "mltk_gecko_sdk_device_efm32gg11b")
add_library(${NAME})
add_library(mltk::gecko_sdk::device_efm32gg11b ALIAS ${NAME})

target_include_directories(${NAME} 
PUBLIC
    Include
)


target_sources(${NAME}  
PRIVATE 
    Source/system_efm32gg11b.c
    Source/startup_efm32gg11b.c
)

mltk_get(SILABS_PART_NUMBER)
if(NOT SILABS_PART_NUMBER)
    mltk_error("Must define variable SILABS_PART_NUMBER, e.g.: mltk_set(SILABS_PART_NUMBER EFM32GG11B820F2048GL192)")
endif()

target_compile_definitions(${NAME} 
PUBLIC 
    __START=_start
    __PROGRAM_START=_dummy
    ${SILABS_PART_NUMBER}
)