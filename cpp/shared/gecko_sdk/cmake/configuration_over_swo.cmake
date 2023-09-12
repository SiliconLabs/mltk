set(NAME "mltk_gecko_sdk_configuration_over_swo")
add_library(${NAME})
add_library(mltk::gecko_sdk::sdk_configuration_over_swo ALIAS ${NAME})

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
  src/sl_cos.c
)

target_link_libraries(${NAME}
PRIVATE
  ${GECKO_SDK_BOARD_TARGET}
  mltk::gecko_sdk::debug_swo
)

