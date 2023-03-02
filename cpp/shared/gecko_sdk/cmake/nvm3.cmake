set(NAME "mltk_gecko_sdk_nvm3")


mltk_get(GECKO_SDK_BOARD_TARGET)
if(NOT GECKO_SDK_BOARD_TARGET)
    mltk_error("Must specify GECKO_SDK_BOARD_TARGET global property")
endif()

mltk_get(CMSIS_CORE)
if("${CMSIS_CORE}" STREQUAL "cortex-m4")
    set(lib_core_name "CM4")
elseif("${CMSIS_CORE}" STREQUAL "cortex-m33")
    set(lib_core_name "CM33")
elseif("${CMSIS_CORE}" STREQUAL "cortex-m3")
    set(lib_core_name "CM3")
else()
    mltk_debug("GSDK NVM3: CMSIS_CORE unsupported: ${CMSIS_CORE}")
    add_library(${NAME} INTERFACE)
    add_library(mltk::gecko_sdk::nvm3 ALIAS ${NAME})
    return()
endif()

add_library(${NAME})
add_library(mltk::gecko_sdk::nvm3 ALIAS ${NAME})

target_include_directories(${NAME}
PUBLIC
    inc
    ${CMAKE_CURRENT_LIST_DIR}/../common/inc
)


target_sources(${NAME}
PRIVATE
  src/nvm3_default_common_linker.c
  src/nvm3_default.c
  src/nvm3_hal_flash.c
  src/nvm3_lock.c
)

target_link_options(${NAME}
PUBLIC
    -Wl,-unvm3_lockBegin
    -Wl,-unvm3_maxFragmentCount
    -Wl,-unvm3_objHandleSize
)

target_link_libraries(${NAME}
PRIVATE
  ${GECKO_SDK_BOARD_TARGET}
  "${CMAKE_CURRENT_LIST_DIR}/../../../../lib/libnvm3_${lib_core_name}_gcc.a"
)