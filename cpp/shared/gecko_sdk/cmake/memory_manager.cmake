
set(NAME "mltk_gecko_sdk_memory_manager")
add_library(${NAME})
add_library(mltk::gecko_sdk::memory_manager ALIAS ${NAME})


mltk_get(GECKO_SDK_BOARD_TARGET)
if(NOT GECKO_SDK_BOARD_TARGET)
    mltk_error("Must specify GECKO_SDK_BOARD_TARGET global property")
endif()


target_include_directories(${NAME} 
PUBLIC
    .
)


target_sources(${NAME}  
PRIVATE 
    sl_malloc.c 
)

target_link_libraries(${NAME}  
PRIVATE
    ${GECKO_SDK_BOARD_TARGET}
)