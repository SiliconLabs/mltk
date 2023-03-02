project(mltk_platform_silabs_common
    VERSION 1.0.0
    DESCRIPTION "Common libs for Silab's platform"
)
export (PACKAGE ${PROJECT_NAME})


add_library(${PROJECT_NAME})
add_library(mltk::platform::silabs_common ALIAS ${PROJECT_NAME})

target_sources(${PROJECT_NAME}
PRIVATE
  microsecond_timer.c
)


target_link_libraries(${PROJECT_NAME}
PRIVATE
  ${MLTK_PLATFORM}
)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/bootloader)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/microphone)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/fault_handler)
