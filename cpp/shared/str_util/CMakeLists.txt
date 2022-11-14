project(mltk_str_util
        VERSION 1.0.0
        DESCRIPTION "String utilities"
)
export (PACKAGE ${PROJECT_NAME})


add_library(${PROJECT_NAME})
add_library(mltk::str_util ALIAS ${PROJECT_NAME})

target_sources(${PROJECT_NAME} 
PRIVATE 
    str_util.c
)

target_include_directories(${PROJECT_NAME}
PUBLIC 
    ${PROJECT_SOURCE_DIR}
)