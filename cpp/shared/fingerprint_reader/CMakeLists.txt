
project(mltk_fingerprint_reader
VERSION 1.0.0
DESCRIPTION "MLTK Fingerprint Reader Driver"
)
export (PACKAGE ${PROJECT_NAME})

add_library(${PROJECT_NAME})
add_library(mltk::fingerprint_reader ALIAS ${PROJECT_NAME})

find_package(mltk_gecko_sdk REQUIRED)



target_sources(${PROJECT_NAME}
PRIVATE 
    fingerprint_reader/fingerprint_reader.c
    fingerprint_reader/r503_driver.c
)

target_include_directories(${PROJECT_NAME} 
PUBLIC 
    ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(${PROJECT_NAME}
PRIVATE
    ${MLTK_PLATFORM}
    mltk::gecko_sdk::includes
)
