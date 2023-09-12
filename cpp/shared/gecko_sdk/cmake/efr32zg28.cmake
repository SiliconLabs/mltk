set(NAME "mltk_gecko_sdk_device_efr32zg28")
add_library(${NAME})
add_library(mltk::gecko_sdk::device_efr32zg28 ALIAS ${NAME})

target_include_directories(${NAME}
PUBLIC
    Include
)


target_sources(${NAME}
PRIVATE
    Source/system_efr32zg28.c
    Source/startup_efr32zg28.c
)

mltk_get(SILABS_PART_NUMBER)
if(NOT SILABS_PART_NUMBER)
    mltk_error("Must define variable SILABS_PART_NUMBER, e.g.: mltk_set(SILABS_PART_NUMBER EFR32ZG28B322F1024IM68)")
endif()

target_compile_options(${NAME}
PUBLIC
    -mcmse
)

target_compile_definitions(${NAME}
PUBLIC
    __START=_start
    __PROGRAM_START=_dummy
    ${SILABS_PART_NUMBER}
)


target_link_libraries(${NAME}
PRIVATE
    ${MLTK_PLATFORM}
)