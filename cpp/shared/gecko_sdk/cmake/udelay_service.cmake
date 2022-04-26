set(NAME "mltk_gecko_sdk_udelay_service")
add_library(${NAME})
add_library(mltk::gecko_sdk::udelay_service ALIAS ${NAME})


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
    src/sl_udelay.c
    src/sl_udelay_armv6m_gcc.S
)

target_link_libraries(${NAME}
PRIVATE 
    ${GECKO_SDK_BOARD_TARGET}
    mltk::gecko_sdk::platform_common
)