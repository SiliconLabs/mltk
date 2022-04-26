project(mltk_flatbuffers
        VERSION 1.0.0
        DESCRIPTION "Wrapper project for Google Flatbuffers"
        HOMEPAGE_URL "https://github.com/google/flatbuffers"
)
export (PACKAGE ${PROJECT_NAME})
add_library(mltk_flatbuffers INTERFACE)	 
add_library(mltk::flatbuffers ALIAS ${PROJECT_NAME})



CPMAddPackage(
NAME FlatBuffers
GITHUB_REPOSITORY google/flatbuffers 
VERSION 1.12.1
CACHE_VERSION v1.12.1
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




# The Flatbuffers CMakeLists.txt is out-dated and generates the warning:
#
# [cmake]   Policy CMP0048 is not set: project() command manages VERSION variables.
# [cmake]   Run "cmake --help-policy CMP0048" for policy details.  Use the cmake_policy
# [cmake]   command to set the policy and suppress this warning.
#
# Run the following Python patching script to remove the warning
#
# NOTE: Unfortunately, the warning will still be printed the first time
#        the Flatbuffers repo is cloned :(
mltk_load_python()
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} ${MLTK_CPP_UTILS_DIR}/libpatcher.py -i ${FlatBuffers_SOURCE_DIR} -p ${CMAKE_CURRENT_LIST_DIR}/patch_flatbuffers.py -o ${FlatBuffers_SOURCE_DIR}/${PROJECT_NAME}_patch_complete.txt
)
