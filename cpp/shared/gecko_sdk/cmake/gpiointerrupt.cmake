set(NAME "mltk_gecko_sdk_gpiointerrupt")
add_library(${NAME})
add_library(mltk::gecko_sdk::gpiointerrupt ALIAS ${NAME})


mltk_get(GECKO_SDK_BOARD_TARGET)
if(NOT GECKO_SDK_BOARD_TARGET)
    mltk_error("Must specify GECKO_SDK_BOARD_TARGET global property")
endif()


target_include_directories(${NAME} 
PUBLIC
    inc
    ${CMAKE_CURRENT_LIST_DIR}/../common/inc
)


target_sources(${NAME}  
PRIVATE 
  src/gpiointerrupt.c  
)

target_link_libraries(${NAME}
PRIVATE 
  ${GECKO_SDK_BOARD_TARGET}
)