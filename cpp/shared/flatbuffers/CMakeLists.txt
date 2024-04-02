project(mltk_flatbuffers
        VERSION 2.0.8
        DESCRIPTION "Wrapper project for Google Flatbuffers"
        HOMEPAGE_URL "https://github.com/google/flatbuffers"
)
export (PACKAGE ${PROJECT_NAME})
add_library(mltk_flatbuffers INTERFACE)	 
add_library(mltk::flatbuffers ALIAS ${PROJECT_NAME})

 

CPMAddPackage(
NAME FlatBuffers
GITHUB_REPOSITORY google/flatbuffers 
VERSION 23.5.26
CACHE_VERSION v23.5.26
OPTIONS 
    "FLATBUFFERS_BUILD_TESTS OFF"
    "FLATBUFFERS_INSTALL OFF"
    "FLATBUFFERS_BUILD_FLATLIB ON" 
    "FLATBUFFERS_BUILD_FLATC ON" 
    "FLATBUFFERS_STATIC_FLATC ON" 
    "FLATBUFFERS_BUILD_FLATHASH OFF" 
    "FLATBUFFERS_BUILD_GRPCTEST OFF" 
    "FLATBUFFERS_BUILD_SHAREDLIB OFF" 
    "FLATBUFFERS_LIBCXX_WITH_CLANG OFF" 
    "FLATBUFFERS_CODE_SANITIZE OFF"
)


target_include_directories(mltk_flatbuffers 
INTERFACE 
    ${FlatBuffers_SOURCE_DIR}/include
)	 

target_compile_definitions(mltk_flatbuffers
INTERFACE
    FLATBUFFERS_LOCALE_INDEPENDENT=0
    FLATBUFFERS_PREFER_PRINTF=1
)