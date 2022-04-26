set(NAME mltk_gecko_sdk_hfxo_manager_service)
add_library(${NAME})
add_library(mltk::gecko_sdk::hfxo_manager_service ALIAS ${NAME})


mltk_get(GECKO_SDK_BOARD_TARGET)
if(NOT GECKO_SDK_BOARD_TARGET)
    mltk_error("Must specify GECKO_SDK_BOARD_TARGET global property")
endif()

target_include_directories(${NAME} 
PUBLIC
    inc
PRIVATE 
    src
)

target_sources(${NAME} 
PRIVATE 
    src/sl_hfxo_manager.c
    src/sl_hfxo_manager_hal_s2.c 
)

target_link_libraries(${NAME}
PRIVATE 
    ${GECKO_SDK_BOARD_TARGET}
    mltk::gecko_sdk::sleeptimer_service
)