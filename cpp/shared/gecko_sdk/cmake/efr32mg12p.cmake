set(NAME "mltk_gecko_sdk_device_efr32mg12p")
add_library(${NAME})
add_library(mltk::gecko_sdk::device_efr32mg12p ALIAS ${NAME})

target_include_directories(${NAME}
PUBLIC
    Include
)


target_sources(${NAME}
PRIVATE
    Source/system_efr32mg12p.c
    Source/startup_efr32mg12p.c
)

mltk_get(SILABS_PART_NUMBER)
if(NOT SILABS_PART_NUMBER)
    mltk_error("Must define variable SILABS_PART_NUMBER, e.g.: mltk_set(SILABS_PART_NUMBER EFR32MG24A010F1536GM48)")
endif()

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