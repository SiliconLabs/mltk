project(mltk_cpputils
    DESCRIPTION "Common C++ utilities and helpers"
)
export (PACKAGE mltk_cpputils)


add_library(${PROJECT_NAME})
add_library(mltk::cpputils ALIAS ${PROJECT_NAME})

target_sources(${PROJECT_NAME}
PRIVATE 
    cpputils/dict.cc 
    cpputils/linked_list.cc 
    cpputils/list.cc
    cpputils/heap.cc
    cpputils/string.cc
    cpputils/buffer.cc
    cpputils/prng.cc
    cpputils/semver.cc
)

target_include_directories(${PROJECT_NAME}
PUBLIC 
    ${CMAKE_CURRENT_LIST_DIR}
)

target_compile_features(${PROJECT_NAME}  PUBLIC cxx_std_17)

