project(mltk_platform_common
        VERSION 1.0.0
        DESCRIPTION "MLTK platform common library"
)
export(PACKAGE mltk_platform_common)


add_library(${PROJECT_NAME})
add_library(mltk::platform::common ALIAS ${PROJECT_NAME})


target_include_directories(${PROJECT_NAME} 
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(${PROJECT_NAME} 
PRIVATE 
    microsecond_timer.c
)

target_link_libraries(${PROJECT_NAME} 
PRIVATE 
    ${MLTK_PLATFORM}
)