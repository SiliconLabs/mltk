project(mltk_stacktrace
        VERSION 1.0.0
        DESCRIPTION "Pretty prints the stacktrace to the console on an unhandled fault/exception"
)
export (PACKAGE ${PROJECT_NAME})


add_library(${PROJECT_NAME})
add_library(mltk::stacktrace ALIAS ${PROJECT_NAME})

target_sources(${PROJECT_NAME} 
PRIVATE 
    stacktrace/stacktrace.c
)

target_include_directories(${PROJECT_NAME}
PUBLIC 
    ${PROJECT_SOURCE_DIR}
)

if(CMAKE_HOST_SYSTEM_NAME STREQUAL Windows)
    target_link_libraries(${PROJECT_NAME} imagehlp)
endif()


add_subdirectory(tests)
