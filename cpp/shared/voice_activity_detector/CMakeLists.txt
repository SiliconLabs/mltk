
# See https://github.com/wiseman/py-webrtcvad
CPMAddPackage(
    NAME py_webrtc_vad 
    GITHUB_REPOSITORY wiseman/py-webrtcvad
    GIT_TAG 2.0.10
    CACHE_VERSION 2.0.10
    CACHE_SUBDIR voice_activity_detector
    DOWNLOAD_ONLY
)
export(PACKAGE mltk_voice_activity_detector)


add_library(mltk_voice_activity_detector)
add_library(mltk::voice_activity_detector ALIAS mltk_voice_activity_detector)





set(py_webrtc_vad_sources 
    signal_processing/complex_bit_reverse.c 
    signal_processing/complex_fft.c 
    signal_processing/cross_correlation.c 
    signal_processing/division_operations.c 
    signal_processing/downsample_fast.c 
    signal_processing/energy.c 
    signal_processing/get_scaling_square.c 
    signal_processing/min_max_operations.c 
    signal_processing/real_fft.c 
    signal_processing/resample_48khz.c
    signal_processing/resample_by_2_internal.c 
    signal_processing/resample_fractional.c 
    signal_processing/vector_scaling_operations.c 
    vad/vad_core.c 
    vad/vad_filterbank.c 
    vad/vad_gmm.c 
    vad/vad_sp.c 
    vad/webrtc_vad.c 
)

set(py_webrtc_vad_inclucdes
    signal_processing/include
    vad/include
)


list(TRANSFORM py_webrtc_vad_sources PREPEND ${py_webrtc_vad_SOURCE_DIR}/cbits/webrtc/common_audio/)
list(TRANSFORM py_webrtc_vad_inclucdes PREPEND ${py_webrtc_vad_SOURCE_DIR}/cbits/webrtc/common_audio/)

target_sources(mltk_voice_activity_detector
PRIVATE 
    voice_activity_detector.c
    ${py_webrtc_vad_sources}
)

target_include_directories(mltk_voice_activity_detector
PUBLIC 
    ${CMAKE_CURRENT_LIST_DIR}
PRIVATE 
    ${py_webrtc_vad_SOURCE_DIR}/cbits
    ${py_webrtc_vad_inclucdes}  
)