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
if(${GTest_SOURCE_DIR} MATCHES "${MLTK_CPP_DIR}/.*")

    mltk_load_python()

    # Ensure the downloaded library is patched 
    add_custom_command(OUTPUT ${GTest_SOURCE_DIR}/${PROJECT_NAME}_patch_complete.txt
        DEPENDS ${GTest_SOURCE_DIR}/googletest ${CMAKE_CURRENT_LIST_DIR}/patch_gtest.py
        COMMAND ${PYTHON_EXECUTABLE} ${MLTK_CPP_UTILS_DIR}/libpatcher.py -i ${GTest_SOURCE_DIR}/googletest -p ${CMAKE_CURRENT_LIST_DIR}/patch_gtest.py -o ${GTest_SOURCE_DIR}/${PROJECT_NAME}_patch_complete.txt
    )
    add_custom_target(${PROJECT_NAME}_apply_patch DEPENDS ${GTest_SOURCE_DIR}/${PROJECT_NAME}_patch_complete.txt)
    add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_apply_patch)

endif()