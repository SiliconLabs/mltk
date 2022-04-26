set(NAME mltk_gecko_sdk_power_manager_service)
add_library(${NAME})
add_library(mltk::gecko_sdk::power_manager_service ALIAS ${NAME})


mltk_get(GECKO_SDK_BOARD_TARGET)
if(NOT GECKO_SDK_BOARD_TARGET)
    mltk_error("Must specify GECKO_SDK_BOARD_TARGET global property")
endif()
mltk_get(GECKO_SDK_BOARD_SERIES)
if(NOT GECKO_SDK_BOARD_SERIES)
    mltk_error("Must specify GECKO_SDK_BOARD_SERIES global property")
endif()

target_include_directories(${NAME} 
PUBLIC
    inc
PRIVATE 
    src
)


if(${GECKO_SDK_BOARD_SERIES} STREQUAL 1)
    set(_hal_source src/sl_power_manager_hal_s0_s1.c)
elseif(${GECKO_SDK_BOARD_SERIES} STREQUAL 2)
    set(_hal_source src/sl_power_manager_hal_s2.c)
else()
    mltk_error("GECKO_SDK_BOARD_SERIES=${GECKO_SDK_BOARD_SERIES} not supported")
endif()


target_sources(${NAME}
PRIVATE 
    src/sl_power_manager.c
    src/sl_power_manager_debug.c
    ${_hal_source}
)

target_compile_options(${NAME}
PRIVATE 
    -Wno-implicit-function-declaration
)


target_link_libraries(${NAME}
PRIVATE 
    ${GECKO_SDK_BOARD_TARGET}
    mltk::gecko_sdk::sleeptimer_service
)