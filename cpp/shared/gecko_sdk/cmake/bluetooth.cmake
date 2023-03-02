set(NAME "mltk_gecko_sdk_bluetooth")
add_library(${NAME})
add_library(mltk::gecko_sdk::bluetooth ALIAS ${NAME})


mltk_get(GECKO_SDK_BOARD_TARGET)
if(NOT GECKO_SDK_BOARD_TARGET)
    mltk_error("Must specify GECKO_SDK_BOARD_TARGET global property")
endif()


target_link_libraries(${NAME}
INTERFACE
    mltk::gecko_sdk::app_bluetooth
PUBLIC
    mltk::gecko_sdk::freertos
    mltk::gecko_sdk::rail
    mltk::gecko_sdk::psa_crypto
PRIVATE
    ${GECKO_SDK_BOARD_TARGET}
)

target_sources(${NAME}
PRIVATE
    src/sl_bt_rtos_adaptation.c
    src/sl_bt_stack_init.c
    src/sli_bt_advertiser_config.c
    src/sli_bt_channel_sounding_config.c
    src/sli_bt_connection_config.c
    src/sli_bt_dynamic_gattdb_config.c
    src/sli_bt_l2cap_config.c
    src/sli_bt_pawr_advertiser_config.c
    src/sli_bt_periodic_adv_config.c
    src/sli_bt_periodic_advertiser_config.c
    src/sli_bt_sync_config.c
)

target_include_directories(${NAME}
PUBLIC
    inc
)


add_library(${NAME}_efr32mg24 INTERFACE)
add_library(mltk::gecko_sdk::bluetooth_efr32mg24 ALIAS ${NAME}_efr32mg24)


target_link_libraries(${NAME}_efr32mg24
INTERFACE
    ${NAME}
    mltk::gecko_sdk::rail_efr32mg24
    ${GECKO_SDK_BOARD_TARGET}
    "${CMAKE_CURRENT_LIST_DIR}/../../../lib/libbluetooth_efr32mg24_gcc.a"
)

target_link_options(${NAME}
PUBLIC
    -Wl,-ustatic_gattdb
)