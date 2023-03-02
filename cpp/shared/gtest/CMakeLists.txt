project(mltk_gtest
        VERSION 1.0.0
        DESCRIPTION "MLTK Google Tests project"
)
export (PACKAGE ${PROJECT_NAME})
add_library(${PROJECT_NAME} INTERFACE)
add_library(mltk::gtest ALIAS ${PROJECT_NAME})


CPMAddPackage(
NAME GTest
GITHUB_REPOSITORY google/googletest
GIT_TAG release-1.11.0
CACHE_VERSION v1.11.0
OPTIONS
    "MSVC OFF" # This is causing bugs when building with SystemC
    "BUILD_GMOCK OFF"
    "INSTALL_GTEST OFF"
    "gtest_disable_pthreads ON"
)



target_link_libraries(${PROJECT_NAME}
INTERFACE
    gtest
)

target_include_directories(${PROJECT_NAME}
INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}
)

# Patch the gtest library if we're using the one that comes with the MLTK
mltk_get(MLTK_PATCH_GTEST_ENABLED)
if(${GTest_SOURCE_DIR} MATCHES "${MLTK_CPP_DIR}/.*" OR MLTK_PATCH_GTEST_ENABLED)
    mltk_load_python()

    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} ${MLTK_CPP_UTILS_DIR}/libpatcher.py -i ${GTest_SOURCE_DIR}/googletest -p ${CMAKE_CURRENT_LIST_DIR}/patch_gtest.py -o ${GTest_SOURCE_DIR}/${PROJECT_NAME}_patch_complete.txt
    )

endif()