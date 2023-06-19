

set(AUDIO_CLASSIFIER_SUPPORTED_PLATFORMS windows linux brd2204 brd2601)
if(NOT ${MLTK_PLATFORM_NAME} IN_LIST AUDIO_CLASSIFIER_SUPPORTED_PLATFORMS)
    mltk_info(
        "Audio Classifier app NOT currently supported by platform: ${MLTK_PLATFORM_NAME}"
        TAG mltk_audio_classifier
    )
    return()
endif()



# Find the necessary packages
find_package(mltk_logging REQUIRED)
find_package(mltk_profiling REQUIRED)
find_package(mltk_tflite_micro_model REQUIRED)
find_package(mltk_gecko_sdk REQUIRED)
find_package(mltk_gecko_sdk_audio_feature_generation REQUIRED)


#####################################################
# Define the audio_classifier executable
add_executable(mltk_audio_classifier)


target_sources(mltk_audio_classifier
PRIVATE
    main.c
    app.c
    audio_classifier.cc
    recognize_commands.cc
    audio_classifier_config.cc
    cli_opts.cc
)

target_link_libraries(mltk_audio_classifier
PRIVATE
    mltk::tflite_micro_model
    mltk::tflite_model_parameters
    mltk::gecko_sdk::audio_feature_generation
    ${MLTK_PLATFORM}
)


target_include_directories(mltk_audio_classifier
PRIVATE
    .
)

mltk_get(MLTK_PLATFORM_IS_EMBEDDED)
if(NOT MLTK_PLATFORM_IS_EMBEDDED)
    find_package(mltk_cxxopts REQUIRED)
    target_link_libraries(mltk_audio_classifier
    PRIVATE
        mltk::cxxopts
    )
endif()


unset(_defines)
mltk_get(VERBOSE)
if(VERBOSE)
    mltk_info("Enabling verbose logs" TAG mltk_audio_classifier)
    set(_defines ${_defines} VERBOSE=1)
endif()

mltk_get(WINDOW_MS)
if(WINDOW_MS)
    mltk_info("WINDOW_MS=${WINDOW_MS}" TAG mltk_audio_classifier)
    set(_defines ${_defines} WINDOW_MS=${WINDOW_MS})
endif()

mltk_get(THRESHOLD)
if(THRESHOLD)
    mltk_info("THRESHOLD=${THRESHOLD}" TAG mltk_audio_classifier)
    set(_defines ${_defines} THRESHOLD=${THRESHOLD})
endif()

mltk_get(SUPPRESSION_MS)
if(SUPPRESSION_MS)
    mltk_info("SUPPRESSION_MS=${SUPPRESSION_MS}" TAG mltk_audio_classifier)
    set(_defines ${_defines} SUPPRESSION_MS=${SUPPRESSION_MS})
endif()

mltk_get(COUNT)
if(COUNT)
    mltk_info("COUNT=${COUNT}" TAG mltk_audio_classifier)
    set(_defines ${_defines} COUNT=${COUNT})
endif()

mltk_get(VOLUME_GAIN)
if(VOLUME_DB)
    mltk_info("VOLUME_GAIN=${VOLUME_GAIN}" TAG mltk_audio_classifier)
    set(_defines ${_defines} VOLUME_GAIN=${VOLUME_GAIN})
endif()

mltk_get(LATENCY_MS)
if(LATENCY_MS)
    mltk_info("LATENCY_MS=${LATENCY_MS}" TAG mltk_audio_classifier)
    set(_defines ${_defines} LATENCY_MS=${LATENCY_MS})
endif()

if(_defines)
    target_compile_definitions(mltk_audio_classifier
    PRIVATE
        ${_defines}
    )
endif()


#####################################################
# Convert the model .tflite to a C array

# Use the default model if one wasn't provided on the command-line
mltk_get(AUDIO_CLASSIFIER_MODEL)
if(NOT AUDIO_CLASSIFIER_MODEL)
    set(AUDIO_CLASSIFIER_MODEL "keyword_spotting_on_off_v3")
endif()
mltk_info("AUDIO_CLASSIFIER_MODEL=${AUDIO_CLASSIFIER_MODEL}" TAG mltk_audio_classifier)

mltk_add_tflite_model(mltk_audio_classifier ${AUDIO_CLASSIFIER_MODEL})

# Generate the exe output files (if necessary for the build platform)
mltk_add_exe_targets(mltk_audio_classifier)



#####################################################
# Optional support for the Audio I/O

mltk_get(AUDIO_CLASSIFIER_ENABLE_AUDIO_IO)
if(AUDIO_CLASSIFIER_ENABLE_AUDIO_IO)
    set(AUDIO_CLASSIFIER_AUDIO_IO_SUPPORTED_PLATFORMS brd2601)
    if(NOT ${MLTK_PLATFORM_NAME} IN_LIST AUDIO_CLASSIFIER_AUDIO_IO_SUPPORTED_PLATFORMS)
        mltk_info(
            "Audio Classifier: Audio I/O support in the is NOT currently supported by platform: ${MLTK_PLATFORM_NAME}"
            TAG mltk_audio_classifier
        )
        return()
    endif()

    mltk_info("Audio Classifier: Enabling audio I/O" TAG mltk_audio_classifier)

    find_package(mltk_opus REQUIRED)
    find_package(mltk_minimp3 REQUIRED)
    find_package(mltk_voice_activity_detector REQUIRED)

    target_link_libraries(mltk_audio_classifier
    PRIVATE
        mltk_opus
        mltk_minimp3
        mltk_voice_activity_detector
    )

    target_sources(mltk_audio_classifier
    PRIVATE
        audio_io.cc
    )

    target_compile_definitions(mltk_audio_classifier
    PRIVATE
        AUDIO_IO_ENABLED=1
    )

endif()



