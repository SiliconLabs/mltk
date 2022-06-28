
add_library(mltk_gecko_sdk_clib_wrappers STATIC)
add_library(mltk::gecko_sdk::clib_wrappers ALIAS mltk_gecko_sdk_clib_wrappers)



find_package(mltk_cpputils)



target_sources(mltk_gecko_sdk_clib_wrappers
PRIVATE 
    clib_wrappers.c
)
target_link_libraries(mltk_gecko_sdk_clib_wrappers
PRIVATE 
    mltk::cpputils
    mltk::gecko_sdk::platform_common
    ${MLTK_PLATFORM}
)


target_link_options(mltk_gecko_sdk_clib_wrappers
INTERFACE 
    -Wl,--wrap,malloc 
    -Wl,--wrap,_malloc_r 
    -Wl,--wrap,calloc 
    -Wl,--wrap,_calloc_r 
    -Wl,--wrap,free 
    -Wl,--wrap,_free_r
    -Wl,--whole-archive ${CMAKE_CURRENT_BINARY_DIR}/libmltk_gecko_sdk_clib_wrappers.a -Wl,--no-whole-archive
)