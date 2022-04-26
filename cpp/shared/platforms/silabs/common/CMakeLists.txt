project(mltk_platform_silabs_common
    VERSION 1.0.0
    DESCRIPTION "Common libs for Silab's platform"
)
export (PACKAGE ${PROJECT_NAME})

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/bootloader)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/microphone)