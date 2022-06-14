
# Only include this example app for Windows/Linux
mltk_get(MLTK_PLATFORM_IS_EMBEDDED)
if(MLTK_PLATFORM_IS_EMBEDDED)
    return()
endif()

set(NAME mltk_microfrontend_example)

add_executable(${NAME})


find_package(mltk_microfrontend REQUIRED)

target_compile_features(${NAME}  PUBLIC cxx_constexpr cxx_std_17)

target_sources(${NAME}
PRIVATE 
    ${CMAKE_CURRENT_LIST_DIR}/../microfrontend/lib/frontend_main.c
)

target_link_libraries(${NAME}
PRIVATE 
    ${MLTK_PLATFORM}
    mltk::microfrontend
)
