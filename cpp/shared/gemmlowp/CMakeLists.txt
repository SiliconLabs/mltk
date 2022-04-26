project(mltk_gemmlowp
        VERSION 1.0.0
        DESCRIPTION "MLTK gemmlowp project"
)
export (PACKAGE ${PROJECT_NAME})
add_library(${PROJECT_NAME} INTERFACE)	 
add_library(mltk::gemmlowp ALIAS ${PROJECT_NAME})


# URL comes from:
# https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/tools/make/third_party_downloads.inc
CPMAddPackage(
    NAME gemmlowp
    URL  https://github.com/google/gemmlowp/archive/719139ce755a0f31cbf1c37f7f98adcc7fc9f425.zip
    URL_HASH MD5=7e8191b24853d75de2af87622ad293ba
    CACHE_VERSION sept2018
)


target_include_directories(${PROJECT_NAME} 
INTERFACE 
    ${gemmlowp_SOURCE_DIR}
)	 
