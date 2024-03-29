project(mltk_platform_linux
        VERSION 1.0.0
        DESCRIPTION "MLTK Linux Platform"
)
export (PACKAGE ${PROJECT_NAME})
add_library(${PROJECT_NAME} INTERFACE)
add_library(mltk::platform::linux ALIAS ${PROJECT_NAME})


mltk_set(MLTK_PLATFORM_NAME linux)
mltk_set(MLTK_PLATFORM_IS_EMBEDDED OFF)
# Default simulated clock rate is 80MHz
mltk_set(MLTK_CPU_CLOCK 80000000 DEFAULT)

# Just force these Gecko SDK values
mltk_set(GECKO_SDK_BOARD_TARGET ${MLTK_PLATFORM})
mltk_set(GECKO_SDK_BOARD_SERIES 2)

find_package(mltk_platform_common REQUIRED)
find_package(mltk_stacktrace REQUIRED)
find_package(mltk_cpputils REQUIRED)
find_package(mltk_libsoundio REQUIRED)
find_package(mltk_gecko_sdk_emulation REQUIRED)

mltk_set(force_link_glibc_2_17_h_path "${CMAKE_CURRENT_LIST_DIR}/force_link_glibc_2.17.h")
function(mltk_platform_linux_link_legacy_glibc)
    # Ensure we link against glibc 2.17 which is what Ubuntu 16
    # For more details, see: https://github.com/wheybags/glibc_version_header
    mltk_get(force_link_glibc_2_17_h_path)
    mltk_warn("Force linking to glibc 2.17 at ${force_link_glibc_2_17_h_path}")
    mltk_append_global_cxx_flags("-include ${force_link_glibc_2_17_h_path}")
endfunction()


add_library(platform_linux_clib_override STATIC)
target_compile_features(platform_linux_clib_override  PUBLIC cxx_constexpr cxx_std_17)
target_sources(platform_linux_clib_override
PRIVATE
    start.cc
    clib_override.c
    sl_sleeptimer_linux.c
    microsecond_timer.c
    sl_system_init.c
)

target_compile_definitions(platform_linux_clib_override
PRIVATE
    MLTK_CPU_CLOCK=${MLTK_CPU_CLOCK}
)
target_link_options(platform_linux_clib_override
PUBLIC
    -Wl,--wrap,main
    -Wl,--wrap,_assert
    -Wl,--wrap,__assert_fail
    -Wl,--wrap,__assert_perror_fail
    -Wl,--wrap,abort
    -Wl,--whole-archive ${CMAKE_CURRENT_BINARY_DIR}/libplatform_linux_clib_override.a -Wl,--no-whole-archive
)
target_link_libraries(platform_linux_clib_override
PUBLIC
    mltk::platform::common
    mltk::stacktrace
    mltk::cpputils
    mltk::libsoundio
    mltk::gecko_sdk::emulation
)

# This ensures ./assert.h (which supports stack tracing)
# is accessible to all files in the build
mltk_append_global_cxx_flags("-I${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(${PROJECT_NAME}
INTERFACE
    mltk::platform::common
    platform_linux_clib_override
    mltk::gecko_sdk::emulation
)

target_compile_definitions(${PROJECT_NAME}
INTERFACE
    __PROGRAM_START=_dummy # This is required to work-aroud build errors in cmsis_gcc.h
)