project(mltk_platform_brd2601
        VERSION 1.0.0
        DESCRIPTION "Silab's BRD2601 Dev board Platform"
)
export (PACKAGE ${PROJECT_NAME})
add_library(${PROJECT_NAME})


#############################################
# MLTK Settings
mltk_set(MLTK_PLATFORM_NAME brd2601)
mltk_set(MLTK_PLATFORM_IS_EMBEDDED ON)

#############################################
# CMSIS Settings
mltk_set(CMSIS_CORE cortex-m33)
mltk_set(CMSIS_FPU fpv4-sp-d16) # should be: fpv5-sp, but not supported by compiler

#############################################
# Gecko SDK Settings
mltk_set(GECKO_SDK_BOARD_SERIES 2)
mltk_set(GECKO_SDK_BOARD_TARGET ${PROJECT_NAME})
mltk_set(GECKO_SDK_INCLUDE_EMLIB ON)
mltk_set(GECKO_SDK_INCLUDE_IOSTREAM_SERVICE ON)
mltk_set(GECKO_SDK_SECURE_ELEMENT_ENABLED ON)

mltk_set(SILABS_PART_NUMBER EFR32MG24B210F1536IM48=1)
# Default clock rate is 78MHz
mltk_set(MLTK_CPU_CLOCK 78000000 DEFAULT)
# Default stack size
mltk_set(MLTK_STACK_SIZE 5120 DEFAULT)
# Default BAUD rate
mltk_set(MLTK_BAUD_RATE 115200 DEFAULT)


target_include_directories(${PROJECT_NAME}
PUBLIC 
    .
    config
    autogen
)

target_sources(${PROJECT_NAME}
PRIVATE
    start.c 
)

add_library(${PROJECT_NAME}_autogen)
target_sources(${PROJECT_NAME}_autogen
PRIVATE 
    autogen/sl_board_default_init.c
    autogen/sl_device_init_clocks.c 
    autogen/sl_event_handler.c 
    autogen/sl_iostream_handles.c 
    autogen/sl_iostream_init_eusart_instances.c 
    autogen/sl_power_manager_handler.c 
    autogen/sl_simple_led_instances.c
    autogen/sl_simple_button_instances.c
)
target_link_libraries(${PROJECT_NAME}_autogen
PRIVATE
    ${PROJECT_NAME}
    mltk::gecko_sdk::iostream_service
    mltk::gecko_sdk::device_init_service
    mltk::gecko_sdk::system_service
    mltk::gecko_sdk::power_manager_service
    mltk::gecko_sdk::hfxo_manager_service
    mltk::gecko_sdk::sleeptimer_service
    mltk::gecko_sdk::udelay_service
    mltk::gecko_sdk::ledrv
    mltk::gecko_sdk::button
    mltk::gecko_sdk::nvm3
    mltk::gecko_sdk::mic
    mltk::gecko_sdk::includes
)
target_compile_options(${PROJECT_NAME}_autogen
PRIVATE 
    -Wno-implicit-function-declaration
)


set(GECKO_SDK_IO_STREAM_SERVICE_SOURCES
    sl_iostream_retarget_stdio.c 
    sl_iostream_uart.c 
    sl_iostream_usart.c 
    sl_iostream_eusart.c 
    sl_iostream_stdlib_config.c
)
mltk_set(GECKO_SDK_IO_STREAM_SERVICE_SOURCES "${GECKO_SDK_IO_STREAM_SERVICE_SOURCES}")




target_link_libraries(${PROJECT_NAME} 
PRIVATE 
    mltk::cpputils 
PUBLIC 
    mltk::gecko_sdk::clib_wrappers
    mltk::platform::common
    mltk::cmsis 
    ${PROJECT_NAME}_api
    mltk::gecko_sdk::iostream_service
    mltk::gecko_sdk::device_init_service
    mltk::gecko_sdk::system_service
    mltk::gecko_sdk::power_manager_service
    mltk::gecko_sdk::sleeptimer_service
    mltk::gecko_sdk::dmadrv
    mltk::gecko_sdk::ledrv
    mltk::gecko_sdk::nvm3
    mltk::gecko_sdk::button
    mltk::gecko_sdk::gpiointerrupt
    mltk::gecko_sdk::emlib
    mltk::gecko_sdk::platform_common
    mltk::gecko_sdk::platform_peripheral
    mltk::gecko_sdk::hardware_board
    mltk::gecko_sdk::mx25_flash_shutdown_eusart
    mltk::gecko_sdk::udelay_service
    mltk::gecko_sdk::device_efr32mg24
    mltk::jlink_stream
    mltk::uart_stream
)

target_link_options(${PROJECT_NAME} 
PUBLIC 
    -Wl,-T${CMAKE_CURRENT_LIST_DIR}/autogen/linkerfile.ld
    -Wl,-u_printf_float # Enable %f in printf()
    -Wl,-u_start
)
set_target_properties(${PROJECT_NAME} PROPERTIES LINK_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/autogen/linkerfile.ld)

target_compile_definitions(${PROJECT_NAME}
PUBLIC 
    SL_COMPONENT_CATALOG_PRESENT
    MLTK_CPU_CLOCK=${MLTK_CPU_CLOCK}
    MLTK_BAUD_RATE=${MLTK_BAUD_RATE}
PRIVATE 
    MLTK_STACK_SIZE=${MLTK_STACK_SIZE}
)

#############################################
# Create the MLTK platform_api library
add_library(${PROJECT_NAME}_api STATIC)
target_sources(${PROJECT_NAME}_api
PRIVATE 
    hardfault_handler.c
    uart_stream_internal.c
)
target_link_options(${PROJECT_NAME}_api
INTERFACE 
    -Wl,--whole-archive ${CMAKE_CURRENT_BINARY_DIR}/lib${PROJECT_NAME}_api.a -Wl,--no-whole-archive
)

target_link_libraries(${PROJECT_NAME}_api
PRIVATE 
    ${PROJECT_NAME}
    mltk::uart_stream
)



###########################################################
# mltk_platform_add_exe_targets
#
# Adds a target to download the application binary to the embedded device
#
macro(mltk_platform_add_exe_targets exe_target)
  mltk_commander_add_download_target(${exe_target})
  mltk_commander_add_reset_target(${exe_target})
  mltk_commander_add_download_run_target(${exe_target} ${ARGN})
endmacro()



#############################################
# Find necessary packages
find_package(mltk_platform_common REQUIRED)
find_package(mltk_platform_silabs_common REQUIRED)
find_package(mltk_cmsis REQUIRED)
find_package(mltk_cpputils REQUIRED)
find_package(mltk_commander REQUIRED)
find_package(mltk_gecko_sdk REQUIRED)
find_package(mltk_jlink_stream REQUIRED)
find_package(mltk_uart_stream REQUIRED)


#############################################
# Enable additional GSDK components

mltk_get(GECKO_SDK_ENABLE_BLUETOOTH)
if(GECKO_SDK_ENABLE_BLUETOOTH)
    mltk_info("Enabling Gecko SDK Bluetooth component")
    mltk_set(GECKO_SDK_ENABLE_FREERTOS ON) # Bluetooth requires the FreeRTOS component

    target_link_libraries(${PROJECT_NAME}
    PUBLIC
        mltk::gecko_sdk::bluetooth_efr32mg24
    )
    target_compile_definitions(${PROJECT_NAME}
    PUBLIC 
        SL_CATALOG_BLUETOOTH_PRESENT
    )

    target_sources(${PROJECT_NAME}_autogen
    PRIVATE 
        autogen/sl_bluetooth.c
    )
endif()

mltk_get(GECKO_SDK_ENABLE_FREERTOS)
if(GECKO_SDK_ENABLE_FREERTOS)
    mltk_info("Enabling Gecko SDK FreeRTOS component")
    target_link_libraries(${PROJECT_NAME}
    PUBLIC
        mltk::gecko_sdk::freertos
    )
    target_compile_definitions(${PROJECT_NAME}
    PUBLIC 
        SL_CATALOG_FREERTOS_KERNEL_PRESENT
    )
endif()


