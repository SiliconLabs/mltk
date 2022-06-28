set(NAME "mltk_gecko_sdk_rail")
add_library(${NAME})
add_library(mltk::gecko_sdk::rail ALIAS ${NAME})


mltk_get(GECKO_SDK_BOARD_TARGET)
if(NOT GECKO_SDK_BOARD_TARGET)
    mltk_error("Must specify GECKO_SDK_BOARD_TARGET global property")
endif()


target_compile_definitions(${NAME}
PUBLIC 
    SL_RAIL_LIB_MULTIPROTOCOL_SUPPORT=0 
    SL_RAIL_UTIL_PA_CONFIG_HEADER=<sl_rail_util_pa_config.h>
    SLI_RADIOAES_REQUIRES_MASKING=1
)

target_sources(${NAME}
PRIVATE
    plugin/rail_util_pti/sl_rail_util_pti.c
)

target_include_directories(${NAME}
PUBLIC
    common
    plugin/pa-conversions
    plugin/rail_util_pti
    protocol/ble
)

target_link_libraries(${NAME}
PRIVATE
    ${GECKO_SDK_BOARD_TARGET}
)





add_library(${NAME}_efr32mg24)
add_library(mltk::gecko_sdk::rail_efr32mg24 ALIAS ${NAME}_efr32mg24)

target_sources(${NAME}_efr32mg24
PRIVATE
    plugin/pa-conversions/pa_conversions_efr32.c
    plugin/pa-conversions/pa_curves_efr32.c
)

target_include_directories(${NAME}_efr32mg24
PUBLIC
    chip/efr32/efr32xg2x
    plugin/pa-conversions/efr32xg24
)

target_link_libraries(${NAME}_efr32mg24
PUBLIC
    ${NAME}
PRIVATE 
  ${GECKO_SDK_BOARD_TARGET}
  "${CMAKE_CURRENT_LIST_DIR}/../../../../lib/librail_efr32xg24_gcc_release.a"
)