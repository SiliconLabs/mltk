
set(NAME "mltk_gecko_sdk_debug_swo")
add_library(${NAME})
add_library(mltk::gecko_sdk::debug_swo ALIAS ${NAME})


mltk_get(GECKO_SDK_BOARD_TARGET)
if(NOT GECKO_SDK_BOARD_TARGET)
    mltk_error("Must specify GECKO_SDK_BOARD_TARGET global property")
endif()


target_include_directories(${NAME}
PUBLIC
    inc
)

target_sources(${NAME}
PRIVATE
    src/sl_debug_swo.c
)

target_link_libraries(${NAME}
PRIVATE
    ${GECKO_SDK_BOARD_TARGET}
)

target_link_options(${NAME}
PUBLIC
    -Wl,-usl_debug_swo_enable_itm
    -Wl,-usl_debug_swo_write_u8
)