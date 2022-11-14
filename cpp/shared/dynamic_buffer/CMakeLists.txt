project(mltk_dynamic_buffer
        VERSION 1.0.0
        DESCRIPTION "Utilities for automatically allocating a buffer as it grows"
)
export (PACKAGE ${PROJECT_NAME})


add_library(${PROJECT_NAME})
add_library(mltk::dynamic_buffer ALIAS ${PROJECT_NAME})

target_sources(${PROJECT_NAME} 
PRIVATE 
    dynamic_buffer.c
)

target_include_directories(${PROJECT_NAME}
PUBLIC 
    ${PROJECT_SOURCE_DIR}
)