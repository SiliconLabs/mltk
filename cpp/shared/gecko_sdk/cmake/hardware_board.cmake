set(NAME "mltk_gecko_sdk_hardware_board")
add_library(${NAME})
add_library(mltk::gecko_sdk::hardware_board ALIAS ${NAME})


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
    src/sl_board_control_gpio.c 
    src/sl_board_init.c
)

target_link_options(${NAME}
PUBLIC 
    -Wl,-usl_board_enable_sensor
    -Wl,-usl_board_enable_vcom
)


target_link_libraries(${NAME}  
PRIVATE
    ${GECKO_SDK_BOARD_TARGET}
    ${GECKO_SDK_BOARD_TARGET}_autogen
)