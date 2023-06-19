
# Currently, this app only supports the BRD2601 platform
# The bluetooth GSDK component must be enabled to build this app
mltk_get(GECKO_SDK_ENABLE_BLUETOOTH)
if(NOT GECKO_SDK_ENABLE_BLUETOOTH OR NOT "${MLTK_PLATFORM_NAME}" STREQUAL "brd2601")
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
add_executable(mltk_ble_audio_classifier)


target_sources(mltk_ble_audio_classifier
PRIVATE
    main.c
    app.c
    recognize_commands.cc
    ble_audio_classifier.cc
    ble_audio_classifier_config.cc
    bluetooth.c
    gatt_db.c
    sl_gatt_service_device_information.c
)

target_link_libraries(mltk_ble_audio_classifier
PRIVATE
    mltk::tflite_micro_model
    mltk::tflite_model_parameters
    mltk::gecko_sdk::audio_feature_generation
    ${MLTK_PLATFORM}
)


target_include_directories(mltk_ble_audio_classifier
PRIVATE
    .
)

mltk_get(MLTK_PLATFORM_IS_EMBEDDED)
if(NOT MLTK_PLATFORM_IS_EMBEDDED)
    find_package(mltk_cxxopts REQUIRED)
    target_link_libraries(mltk_ble_audio_classifier
    PRIVATE
        mltk::cxxopts
    )
endif()


#####################################################
# Convert the model .tflite to a C array

# Use the default model if one wasn't provided on the command-line
mltk_get(BLE_AUDIO_CLASSIFIER_MODEL)
if(NOT BLE_AUDIO_CLASSIFIER_MODEL)
    set(BLE_AUDIO_CLASSIFIER_MODEL "keyword_spotting_pacman_v3")
endif()
mltk_info("BLE_AUDIO_CLASSIFIER_MODEL=${BLE_AUDIO_CLASSIFIER_MODEL}" TAG mltk_ble_audio_classifier)

mltk_add_tflite_model(mltk_ble_audio_classifier ${BLE_AUDIO_CLASSIFIER_MODEL})

# Generate the exe output files (if necessary for the build platform)
mltk_add_exe_targets(mltk_ble_audio_classifier)
