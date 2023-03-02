

if(NOT TRANSFORM OR CMSIS_DSP_ENABLE_ALL)
    set(TRANSFORM ON)
endif()
if(NOT CONTROLLER AND NOT CMSIS_DSP_ENABLE_ALL)
    set(CONTROLLER OFF)
endif()
if(NOT COMPLEXMATH AND NOT CMSIS_DSP_ENABLE_ALL)
    set(COMPLEXMATH OFF)
endif()
if(NOT FILTERING AND NOT CMSIS_DSP_ENABLE_ALL)
    set(FILTERING OFF)
endif()
if(NOT MATRIX AND NOT CMSIS_DSP_ENABLE_ALL)
    set(MATRIX OFF)
endif()
if(NOT STATISTICS AND NOT CMSIS_DSP_ENABLE_ALL)
    set(STATISTICS OFF)
endif()
if(NOT SVM AND NOT CMSIS_DSP_ENABLE_ALL)
    set(SVM OFF)
endif()
if(NOT BAYES AND NOT CMSIS_DSP_ENABLE_ALL)
    set(BAYES OFF)
endif()
if(NOT DISTANCE AND NOT CMSIS_DSP_ENABLE_ALL)
    set(DISTANCE OFF)
endif()
if(NOT INTERPOLATION AND NOT CMSIS_DSP_ENABLE_ALL)
    set(INTERPOLATION OFF)
endif()
if(NOT QUATERNIONMATH AND NOT CMSIS_DSP_ENABLE_ALL)
    set(QUATERNIONMATH OFF)
endif()



CPMAddPackage(
    NAME cmsis_dsp
    URL https://github.com/ARM-software/CMSIS-DSP/archive/refs/tags/v1.11.0.zip
    URL_HASH SHA1=db167d876b3ab30e8a35f74e0d6be91746aacb30
    DOWNLOAD_ONLY ON
    CACHE_SUBDIR cmsis
    CACHE_VERSION v1.11
)
set(cmsis_dsp_SOURCE_DIR ${cmsis_dsp_SOURCE_DIR} CACHE INTERNAL "")



add_library(mltk_cmsis_dsp INTERFACE)
add_library(mltk::cmsis_dsp ALIAS mltk_cmsis_dsp)


add_subdirectory("${cmsis_dsp_SOURCE_DIR}/Source")

target_link_libraries(mltk_cmsis_dsp
INTERFACE
    CMSISDSP
    mltk_cmsis
)

target_compile_definitions(mltk_cmsis_dsp
INTERFACE
    CMSIS_FORCE_BUILTIN_FUNCTIONS
)


mltk_load_python()

# Ensure the downloaded library is patched
add_custom_command(OUTPUT ${cmsis_dsp_SOURCE_DIR}/mltk_cmsis_dsp_patch_complete.txt
  DEPENDS ${cmsis_dsp_SOURCE_DIR}/Include ${CMAKE_CURRENT_LIST_DIR}/patch_cmsis_dsp.py
  COMMAND ${PYTHON_EXECUTABLE} ${MLTK_CPP_UTILS_DIR}/libpatcher.py -i "${cmsis_dsp_SOURCE_DIR}/Include" -p ${CMAKE_CURRENT_LIST_DIR}/patch_cmsis_dsp.py -o ${cmsis_dsp_SOURCE_DIR}/mltk_cmsis_dsp_patch_complete.txt
)
add_custom_target(mltk_cmsis_dsp_apply_patch DEPENDS ${cmsis_dsp_SOURCE_DIR}/mltk_cmsis_dsp_patch_complete.txt)
add_dependencies(CMSISDSP mltk_cmsis_dsp_apply_patch)