project(mltk_gecko_sdk_emulation
        VERSION 1.0.0
        DESCRIPTION "MLTK Silabs Gecko SDK API emulation library"
)
export(PACKAGE mltk_gecko_sdk_emulation)
add_library(${PROJECT_NAME})
add_library(mltk::gecko_sdk::emulation ALIAS ${PROJECT_NAME})



find_package(mltk_gecko_sdk REQUIRED)



target_include_directories(${PROJECT_NAME}
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)


target_sources(${PROJECT_NAME}  
PRIVATE
    sl_system_init.c
    sl_simple_led_instances.c
    sl_led.c
    em_cmu.c
)

if(HOST_OS_IS_WINDOWS)
    target_sources(${PROJECT_NAME}  
    PRIVATE
        sl_sleeptimer_windows.c
    )
elseif(HOST_OS_IS_LINUX)
    target_sources(${PROJECT_NAME}  
    PRIVATE
        sl_sleeptimer_linux.c
    )
else()
    mltk_error("Host OS is not currently supported")
endif()


mltk_get(MLTK_CPU_CLOCK)
target_compile_definitions(${PROJECT_NAME}
PRIVATE 
    MLTK_CPU_CLOCK=${MLTK_CPU_CLOCK}
)


target_link_libraries(${PROJECT_NAME}
PUBLIC 
    mltk::gecko_sdk::includes
PRIVATE 
    ${MLTK_PLATFORM}
)
