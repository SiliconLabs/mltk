
set(NAME "mltk_gecko_sdk_freertos")
add_library(${NAME})
add_library(mltk::gecko_sdk::freertos ALIAS ${NAME})



mltk_get(GECKO_SDK_BOARD_TARGET)
if(NOT GECKO_SDK_BOARD_TARGET)
    mltk_error("Must specify GECKO_SDK_BOARD_TARGET global property")
endif()
mltk_get(CMSIS_CORE)
if(NOT CMSIS_CORE)
    mltk_error("Must specify CMSIS_CORE global property")
endif()


find_package(mltk_cmsis REQUIRED)



target_include_directories(${NAME}
PUBLIC
    cmsis/Include
    kernel/include
)

target_sources(${NAME}
PRIVATE
    cmsis/Source/cmsis_os2.c
    kernel/croutine.c
    kernel/event_groups.c
    kernel/list.c
    kernel/queue.c
    kernel/stream_buffer.c
    kernel/tasks.c
    kernel/timers.c
    kernel/portable/MemMang/heap_4.c
    kernel/portable/SiliconLabs/tick_power_manager.c
    ../../../platform/common/src/sli_cmsis_os2_ext_task_register.c
)

target_link_libraries(${NAME}
PRIVATE
    ${GECKO_SDK_BOARD_TARGET}
PUBLIC
    mltk_cmsis_rtos2
)



if(CMSIS_CORE STREQUAL "cortex-m33")
    target_include_directories(${NAME}
    PUBLIC
        kernel/portable/GCC/ARM_CM33_NTZ/non_secure
    )

    target_sources(${NAME}
    PRIVATE
        kernel/portable/GCC/ARM_CM33_NTZ/non_secure/portasm.c
        kernel/portable/GCC/ARM_CM33_NTZ/non_secure/port.c
    )

elseif(CMSIS_CORE STREQUAL "cortex-m4")
    target_include_directories(${NAME}
    PUBLIC
        kernel/portable/GCC/ARM_CM4F
    )

    target_sources(${NAME}
    PRIVATE
        kernel/portable/GCC/ARM_CM4F/port.c
    )
else()
    mltk_debug("GSDK FreeRTOS: CMSIS_CORE unsupported: ${CMSIS_CORE}")
endif()
