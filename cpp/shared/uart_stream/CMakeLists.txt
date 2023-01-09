
project(mltk_uart_stream
  VERSION 1.0.0
  DESCRIPTION "MLTK UART Data Stream Utility"
)
export (PACKAGE ${PROJECT_NAME})

find_package(mltk_cpputils REQUIRED)
find_package(mltk_logging REQUIRED)
find_package(mltk_gecko_sdk REQUIRED)

add_library(${PROJECT_NAME})
add_library(mltk::uart_stream ALIAS ${PROJECT_NAME})


target_sources(${PROJECT_NAME}
PRIVATE 
  src/uart_stream.cc
)

target_include_directories(${PROJECT_NAME} 
PRIVATE 
  src
PUBLIC 
  ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(${PROJECT_NAME}
PRIVATE
  ${MLTK_PLATFORM}
  mltk::cpputils
  mltk::gecko_sdk::dmadrv
  mltk::gecko_sdk::emlib
)


target_link_options(${PROJECT_NAME}
PUBLIC 
  -Wl,-uDMADRV_LdmaStartTransfer
)