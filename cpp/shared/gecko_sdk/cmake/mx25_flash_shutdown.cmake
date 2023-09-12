set(NAME "mltk_gecko_sdk_mx25_flash_shutdown")
add_library(${NAME}_eusart)
add_library(mltk::gecko_sdk::mx25_flash_shutdown_eusart ALIAS ${NAME}_eusart)
add_library(${NAME}_usart)
add_library(mltk::gecko_sdk::mx25_flash_shutdown_usart ALIAS ${NAME}_usart)

mltk_get(GECKO_SDK_BOARD_TARGET)
if(NOT GECKO_SDK_BOARD_TARGET)
    mltk_error("Must specify GECKO_SDK_BOARD_TARGET global property")
endif()


target_include_directories(${NAME}_eusart
PUBLIC
    inc/sl_mx25_flash_shutdown_eusart
)

target_sources(${NAME}_eusart
PRIVATE
  src/sl_mx25_flash_shutdown_eusart/sl_mx25_flash_shutdown.c
)

target_link_libraries(${NAME}_eusart
PRIVATE
  ${GECKO_SDK_BOARD_TARGET}
)


target_include_directories(${NAME}_usart
PUBLIC
    inc/sl_mx25_flash_shutdown_usart
)

target_sources(${NAME}_usart
PRIVATE
  src/sl_mx25_flash_shutdown_usart/sl_mx25_flash_shutdown.c
)

target_link_libraries(${NAME}_usart
PRIVATE
  ${GECKO_SDK_BOARD_TARGET}
)