set(NAME "mltk_gecko_sdk_system_service")
add_library(${NAME})
add_library(mltk::gecko_sdk::system_service ALIAS ${NAME})


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
    src/sl_system_init.c
    src/sl_system_process_action.c
    src/sl_system_kernel.c
)

target_link_libraries(${NAME}
PRIVATE 
    ${GECKO_SDK_BOARD_TARGET}
    mltk::gecko_sdk::platform_common
)