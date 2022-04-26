
project(mltk_jlink_stream
  VERSION 1.0.0
  DESCRIPTION "MLTK JLink Data Stream Utility"
)
export (PACKAGE ${PROJECT_NAME})

find_package(mltk_cpputils REQUIRED)
find_package(mltk_logging REQUIRED)


add_library(${PROJECT_NAME})
add_library(mltk::jlink_stream ALIAS ${PROJECT_NAME})


target_sources(${PROJECT_NAME}
PRIVATE 
  jlink_stream/jlink_stream.cc
  jlink_stream/jlink_stream_command.cc
  jlink_stream/jlink_stream_internal.cc
)

target_include_directories(${PROJECT_NAME} 
PRIVATE 
  .
PUBLIC 
  ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(${PROJECT_NAME}
PRIVATE
  ${MLTK_PLATFORM}
  mltk::cpputils
  mltk::logging
)
