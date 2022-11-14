project(mltk_msgpack
        VERSION 1.0.0
        DESCRIPTION "Utility to serialize and de-serialize data using the msgpack format"
)
export (PACKAGE ${PROJECT_NAME})


add_library(${PROJECT_NAME})
add_library(mltk::msgpack ALIAS ${PROJECT_NAME})

find_package(mltk_str_util REQUIRED)
find_package(mltk_dynamic_buffer REQUIRED)

target_sources(${PROJECT_NAME} 
PRIVATE 
    src/msgpack.c
    src/msgpack_deserialize.c 
    src/msgpack_serialize.c
    src/msgpack_buffered_writer.c
)

target_include_directories(${PROJECT_NAME}
PRIVATE 
    ./src
PUBLIC 
    ${PROJECT_SOURCE_DIR}
)

target_link_libraries(${PROJECT_NAME}
PRIVATE 
    mltk::str_util
    mltk::dynamic_buffer
)