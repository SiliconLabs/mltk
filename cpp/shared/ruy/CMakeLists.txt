project(mltk_ruy
        VERSION 1.0.0
)
export (PACKAGE ${PROJECT_NAME})
add_library(${PROJECT_NAME} INTERFACE)	 
add_library(mltk::ruy ALIAS ${PROJECT_NAME})


# URL comes from:
# https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/tools/make/third_party_downloads.inc
CPMAddPackage(
    NAME ruy
    URL  https://github.com/google/ruy/archive/d37128311b445e758136b8602d1bbd2a755e115d.zip
    URL_HASH MD5=abf7a91eb90d195f016ebe0be885bb6e
    CACHE_VERSION may2021
)
    
    
target_include_directories(${PROJECT_NAME}
INTERFACE 
    ${ruy_SOURCE_DIR}
)	 
