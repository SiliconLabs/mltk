project(mltk_cmsis
        VERSION 1.0.0
        DESCRIPTION "ARM CMSIS Project")
export (PACKAGE ${PROJECT_NAME})
add_library(${PROJECT_NAME} INTERFACE)
add_library(mltk::cmsis ALIAS ${PROJECT_NAME})



mltk_get(CMSIS_CORE)
mltk_get(CMSIS_FPU)
mltk_get(CMSIS_MARCH) # https://gcc.gnu.org/onlinedocs/gcc/ARM-Options.html
mltk_get(CMSIS_DISABLE_FLOAT16)
mltk_get(MLTK_PLATFORM_IS_EMBEDDED)



CPMAddPackage(
    NAME cmsis
    URL  https://github.com/ARM-software/CMSIS_5/archive/a75f01746df18bb5b929dfb8dc6c9407fac3a0f3.zip
    URL_HASH SHA1=81f75c6401e57264a393c9344c255d18f426455e
    DOWNLOAD_ONLY ON
    CACHE_VERSION jun21_2023
)
set(cmsis_SOURCE_DIR ${cmsis_SOURCE_DIR} CACHE INTERNAL "")



if(NOT MLTK_PLATFORM_IS_EMBEDDED)
    include("${CMAKE_CURRENT_LIST_DIR}/wrapper/wrapper_build.cmake")
    return()
endif()


if(NOT CMSIS_CORE)
    mltk_error("Must define global variable CMSIS_CORE" TAG mltk_cmsis)
endif()

set(CMSIS_SUPPORTED_CORES cortex-m3 cortex-m4 cortex-m33 cortex-m55)
if(NOT ${CMSIS_CORE} IN_LIST CMSIS_SUPPORTED_CORES)
    mltk_error("CMSIS_CORE unsupported: ${CMSIS_CORE}\nCMSIS_SUPPORTED_CORES: ${CMSIS_SUPPORTED_CORES}" TAG mltk_cmsis)
endif()


mltk_append_global_cxx_flags("-mthumb -mno-thumb-interwork -flax-vector-conversions -I${cmsis_SOURCE_DIR}/CMSIS/Core/Include" ASM)


mltk_remove_global_cxx_flags("-mfloat-abi=.*" WILDCARD ASM)
mltk_remove_global_cxx_flags("-mfpu=.*" WILDCARD ASM)
if(NOT CMSIS_FPU)
    mltk_info("Using soft FPU" TAG mltk_cmsis)
    mltk_append_global_cxx_flags("-mfloat-abi=softfp" ASM)
else()
    mltk_info("Using hw FPU=${CMSIS_FPU}" TAG mltk_cmsis)
    mltk_append_global_cxx_flags("-mfloat-abi=hard -mfpu=${CMSIS_FPU}" ASM)
endif()

if(CMSIS_DISABLE_FLOAT16)
    mltk_remove_global_cxx_flags("-mfp16-format=ieee" ASM)
    set(DISABLEFLOAT16 ON)
else()
    mltk_info("Enabling float16" TAG mltk_cmsis)
    mltk_append_global_cxx_flags("-mfp16-format=ieee" ASM)
    set(DISABLEFLOAT16 OFF)
endif()


mltk_remove_global_cxx_flags("-march=.*" WILDCARD ASM)
mltk_remove_global_cxx_flags("-mcpu=.*" WILDCARD ASM)
mltk_remove_global_cxx_flags("-mtune=.*" WILDCARD ASM)
if(CMSIS_MARCH)
    mltk_info("Using -march=${CMSIS_MARCH}" TAG mltk_cmsis)
    mltk_append_global_cxx_flags("-march=${CMSIS_MARCH}" ASM)
else()
    mltk_info("Using -mcpu=${CMSIS_CORE}" TAG mltk_cmsis)
    mltk_append_global_cxx_flags("-mcpu=${CMSIS_CORE} -mtune=${CMSIS_CORE}" ASM)
endif()



target_include_directories(${PROJECT_NAME}
INTERFACE
    ${cmsis_SOURCE_DIR}
)

include(${CMAKE_CURRENT_LIST_DIR}/cmsis_dsp.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmsis_nn.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmsis_device.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmsis_rtos2.cmake)



