
# See https://github.com/xiph/opus
CPMAddPackage(
    NAME opus 
    GITHUB_REPOSITORY xiph/opus
    GIT_TAG 757c53f775a0b651b0512a1992d67f4b2159a378
    CACHE_VERSION nov24_2022
    OPTIONS 
        "CMAKE_MESSAGE_LOG_LEVEL NOTICE"
        "OPUS_FIXED_POINT ON"
        "OPUS_ENABLE_FLOAT_API OFF"
        "OPUS_INSTALL_PKG_CONFIG_MODULE OFF"
        "OPUS_INSTALL_CMAKE_CONFIG_MODULE OFF"
        "OPUS_STACK_PROTECTOR OFF"
        "OPUS_BUILD_SHARED_LIBRARY OFF"
        "OPUS_DISABLE_INTRINSICS ON"
)

export(PACKAGE mltk_opus)
add_library(mltk_opus INTERFACE)
add_library(mltk::opus ALIAS mltk_opus)
target_link_libraries(mltk_opus
INTERFACE
    opus
)

