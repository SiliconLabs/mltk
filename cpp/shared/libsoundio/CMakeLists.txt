project(mltk_libsoundio
        VERSION 1.0.0
        DESCRIPTION "C library providing cross-platform audio input and output. "
        HOMEPAGE_URL "https://github.com/andrewrk/libsoundio")
export(PACKAGE ${PROJECT_NAME})
add_library(${PROJECT_NAME})
add_library(mltk::libsoundio ALIAS ${PROJECT_NAME})

CPMAddPackage(
    NAME libsoundio 
    GITHUB_REPOSITORY  andrewrk/libsoundio
    GIT_TAG 2.0.0
    CACHE_VERSION v2.0.0
    DOWNLOAD_ONLY ON
)


# NOTE: This package is causing build issues with other projects
#       so we just manually add the files we want to build


target_include_directories(${PROJECT_NAME}
PUBLIC
  .
  ${libsoundio_BINARY_DIR}
  ${libsoundio_SOURCE_DIR}
)

target_compile_definitions(${PROJECT_NAME}
PUBLIC
  SOUNDIO_STATIC_LIBRARY
)

mltk_discover_host_os()

set(LIBSOUNDIO_VERSION_MAJOR 2)
set(LIBSOUNDIO_VERSION_MINOR 0)
set(LIBSOUNDIO_VERSION_PATCH 0)
set(LIBSOUNDIO_VERSION "2.0.0")
set(SOUNDIO_HAVE_PULSEAUDIO ${HOST_OS_IS_LINUX})
set(SOUNDIO_HAVE_COREAUDIO ${HOST_OS_IS_OSX})
set(SOUNDIO_HAVE_WASAPI ${HOST_OS_IS_WINDOWS})


set(LIBSOUNDIO_SOURCES
    "${libsoundio_SOURCE_DIR}/src/soundio.c"
    "${libsoundio_SOURCE_DIR}/src/util.c"
    "${libsoundio_SOURCE_DIR}/src/os.c"
    "${libsoundio_SOURCE_DIR}/src/dummy.c"
    "${libsoundio_SOURCE_DIR}/src/channel_layout.c"
    "${libsoundio_SOURCE_DIR}/src/ring_buffer.c"
)



if(SOUNDIO_HAVE_PULSEAUDIO)
    find_package(PulseAudio QUIET)

    if(PulseAudio_FOUND)
        target_include_directories(${PROJECT_NAME}
        PRIVATE
            ${PULSEAUDIO_INCLUDE_DIR}
        )

        set(LIBSOUNDIO_SOURCES ${LIBSOUNDIO_SOURCES}
            "${libsoundio_SOURCE_DIR}/src/pulseaudio.c"
        )
    else()
        set(SOUNDIO_HAVE_PULSEAUDIO OFF)
        mltk_warn("Failed to find the PulseAudio library required by the microphone library")
        mltk_warn("Try running: sudo apt install pulseaudio")
    endif()
endif()
if(SOUNDIO_HAVE_COREAUDIO)
    set(LIBSOUNDIO_SOURCES ${LIBSOUNDIO_SOURCES}
        "${libsoundio_SOURCE_DIR}/src/coreaudio.c"
    )
endif()
if(SOUNDIO_HAVE_WASAPI)
    set(LIBSOUNDIO_SOURCES ${LIBSOUNDIO_SOURCES}
        "${libsoundio_SOURCE_DIR}/src/wasapi.c"
    )
endif()

set(CONFIGURE_OUT_FILE "${libsoundio_BINARY_DIR}/config.h")
configure_file(
    "${libsoundio_SOURCE_DIR}/src/config.h.in"
    ${CONFIGURE_OUT_FILE}
)



find_package(mltk_gecko_sdk REQUIRED)
find_package(mltk_cpputils REQUIRED)

target_link_libraries(${PROJECT_NAME}
PRIVATE
    mltk::gecko_sdk::includes
    mltk::cpputils
    ${MLTK_PLATFORM}
)

target_sources(${PROJECT_NAME}
PRIVATE 
  ${LIBSOUNDIO_SOURCES}
  mltk_sl_mic.cc
)

