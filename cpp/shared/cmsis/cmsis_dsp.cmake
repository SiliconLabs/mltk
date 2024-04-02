

if(NOT DEFINED TRANSFORM OR CMSIS_DSP_ENABLE_ALL)
    set(TRANSFORM ON)
endif()
if(NOT DEFINED CONTROLLER AND NOT CMSIS_DSP_ENABLE_ALL)
    set(CONTROLLER OFF)
endif()
if(NOT DEFINED COMPLEXMATH AND NOT CMSIS_DSP_ENABLE_ALL)
    set(COMPLEXMATH OFF)
endif()
if(NOT DEFINED FILTERING AND NOT CMSIS_DSP_ENABLE_ALL)
    set(FILTERING OFF)
endif()
if(NOT DEFINED MATRIX AND NOT CMSIS_DSP_ENABLE_ALL)
    set(MATRIX ON)
endif()
if(NOT DEFINED STATISTICS AND NOT CMSIS_DSP_ENABLE_ALL)
    set(STATISTICS OFF)
endif()
if(NOT DEFINED SVM AND NOT CMSIS_DSP_ENABLE_ALL)
    set(SVM OFF)
endif()
if(NOT DEFINED BAYES AND NOT CMSIS_DSP_ENABLE_ALL)
    set(BAYES OFF)
endif()
if(NOT DEFINED DISTANCE AND NOT CMSIS_DSP_ENABLE_ALL)
    set(DISTANCE OFF)
endif()
if(NOT DEFINED INTERPOLATION AND NOT CMSIS_DSP_ENABLE_ALL)
    set(INTERPOLATION OFF)
endif()
if(NOT DEFINED QUATERNIONMATH AND NOT CMSIS_DSP_ENABLE_ALL)
    set(QUATERNIONMATH OFF)
endif()



CPMAddPackage(
    NAME cmsis_dsp
    URL https://github.com/ARM-software/CMSIS-DSP/archive/refs/tags/v1.15.0.zip
    URL_HASH SHA1=130b5faa006fc0cd8939270b7afd507f2ea3f077
    DOWNLOAD_ONLY ON
    CACHE_SUBDIR cmsis
    CACHE_VERSION v1.15
)
set(cmsis_dsp_SOURCE_DIR ${cmsis_dsp_SOURCE_DIR} CACHE INTERNAL "")



if(MLTK_PLATFORM_IS_EMBEDDED)
    add_subdirectory("${cmsis_dsp_SOURCE_DIR}/Source")

    add_library(mltk_cmsis_dsp INTERFACE)
    add_library(mltk::cmsis_dsp ALIAS mltk_cmsis_dsp)

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

endif() # MLTK_PLATFORM_IS_EMBEDDED