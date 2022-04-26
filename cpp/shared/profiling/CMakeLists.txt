project(mltk_profiling
        VERSION 1.0.0
        DESCRIPTION "Profiling utility"
)
export (PACKAGE ${PROJECT_NAME})


find_package(mltk_cpputils REQUIRED)
find_package(mltk_logging REQUIRED)

add_library(${PROJECT_NAME})
add_library(mltk::profiling ALIAS ${PROJECT_NAME})


target_sources(${PROJECT_NAME}
PRIVATE 
    profiling/profiler.cc
    profiling/profiling.cc
    profiling/profiler_fullname.cc
)

target_include_directories(${PROJECT_NAME} 
PUBLIC 
    ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(${PROJECT_NAME}
PUBLIC 
    ${MLTK_PLATFORM}
    mltk::logging
    mltk::cpputils  
)
