set(NAME "mltk_gecko_sdk_app_bluetooth")
add_library(${NAME} INTERFACE)
add_library(mltk::gecko_sdk::app_bluetooth ALIAS ${NAME})


mltk_get(GECKO_SDK_BOARD_TARGET)
if(NOT GECKO_SDK_BOARD_TARGET)
    mltk_error("Must specify GECKO_SDK_BOARD_TARGET global property")
endif()


target_include_directories(${NAME}
INTERFACE
    common/gatt_service_device_information
    ../common/util/app_assert
)
