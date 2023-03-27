project(mltk_float16
        VERSION 1.0.0
        DESCRIPTION "Float16 data type support"
)
export(PACKAGE mltk_float16)
add_library(${PROJECT_NAME} INTERFACE)
add_library(mltk::float16 ALIAS ${PROJECT_NAME})


target_include_directories(${PROJECT_NAME}
INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}
)


target_include_directories(${PROJECT_NAME}
INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/half/v2.1.0/include
)
target_compile_definitions(${PROJECT_NAME}
INTERFACE
    HALF_ENABLE_CPP11_CFENV=0
)
