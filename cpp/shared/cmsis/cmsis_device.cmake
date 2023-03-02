

if(CMSIS_CORE STREQUAL "cortex-m3")
    set(CMSIS_DEVICE "ARMCM3")
elseif(CMSIS_CORE STREQUAL "cortex-m4")
    set(CMSIS_DEVICE "ARMCM4")
elseif(CMSIS_CORE STREQUAL "cortex-m33")
    set(CMSIS_DEVICE "ARMCM33")
elseif(CMSIS_CORE STREQUAL "cortex-m55")
    set(CMSIS_DEVICE "ARMCM55")
else()
    mltk_error("Unsupported CMSIS_CORE: ${CMSIS_CORE}" TAG mltk_cmsis)
endif()


add_library(mltk_cmsis_device)


target_sources(mltk_cmsis_device
PRIVATE
    ${cmsis_SOURCE_DIR}/Device/ARM/${CMSIS_DEVICE}/Source/startup_${CMSIS_DEVICE}.c
    ${cmsis_SOURCE_DIR}/Device/ARM/${CMSIS_DEVICE}/Source/system_${CMSIS_DEVICE}.c
)


target_include_directories(mltk_cmsis_device
PUBLIC
    ${cmsis_SOURCE_DIR}/Device/ARM/${CMSIS_DEVICE}/Include
    ${cmsis_SOURCE_DIR}/Device/ARM/${CMSIS_DEVICE}/Include
    ${cmsis_SOURCE_DIR}/Device/ARM/${CMSIS_DEVICE}/Include/Template
)