project(mltk_unity
        VERSION 1.0.0
        DESCRIPTION "MLTK Unity Testing Framework"
        HOMEPAGE_URL "https://github.com/ThrowTheSwitch/Unity"
)
export(PACKAGE ${PROJECT_NAME})


CPMAddPackage(
  NAME unity
  GIT_REPOSITORY https://github.com/ThrowTheSwitch/Unity
  GIT_TAG v2.5.2
  CACHE_VERSION 2.5.2
  OPTIONS
    "UNITY_EXTENSION_FIXTURE ON"
)
set(unity_SOURCE_DIR ${unity_SOURCE_DIR} CACHE INTERNAL "")

add_library(mltk_unity)
add_library(mltk::unity ALIAS mltk_unity)
target_link_libraries(mltk_unity
PUBLIC
    unity
)

target_sources(mltk_unity
PRIVATE
  unity_util.c
)

mltk_append_global_cxx_defines(UNITY_INCLUDE_CONFIG_H)
mltk_append_global_cxx_flags("-I${CMAKE_CURRENT_LIST_DIR}")



mltk_load_python()

# Ensure the downloaded library is patched
add_custom_command(OUTPUT ${unity_SOURCE_DIR}/mltk_unity_patch_complete.txt
  DEPENDS ${unity_SOURCE_DIR}/src ${CMAKE_CURRENT_LIST_DIR}/patch_unity.py
  COMMAND ${PYTHON_EXECUTABLE} ${MLTK_CPP_UTILS_DIR}/libpatcher.py -i "${unity_SOURCE_DIR}/src" -p ${CMAKE_CURRENT_LIST_DIR}/patch_unity.py -o ${unity_SOURCE_DIR}/mltk_unity_patch_complete.txt
)
add_custom_target(mltk_unity_apply_patch DEPENDS ${unity_SOURCE_DIR}/mltk_unity_patch_complete.txt)
add_dependencies(mltk_unity mltk_unity_apply_patch)