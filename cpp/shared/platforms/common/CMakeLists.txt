project(mltk_platform_common
        VERSION 1.0.0
        DESCRIPTION "MLTK platform common library"
)
export(PACKAGE mltk_platform_common)


add_library(${PROJECT_NAME} INTERFACE)
add_library(mltk::platform::common ALIAS ${PROJECT_NAME})


target_include_directories(${PROJECT_NAME}
INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
