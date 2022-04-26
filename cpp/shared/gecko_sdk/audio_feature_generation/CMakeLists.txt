project(mltk_gecko_sdk_audio_feature_generation
    VERSION 1.0.0
    DESCRIPTION "MLTK audio feature generation library used by Gecko SDK"
)
export(PACKAGE ${PROJECT_NAME})
add_library(${PROJECT_NAME})
add_library(mltk::gecko_sdk::audio_feature_generation ALIAS ${PROJECT_NAME})

find_package(mltk_microfrontend REQUIRED)
find_package(mltk_cpputils REQUIRED)
find_package(mltk_tflite_micro REQUIRED)
find_package(mltk_tflite_model_parameters REQUIRED)
find_package(mltk_jlink_stream REQUIRED)
find_package(mltk_libsoundio REQUIRED)


mltk_get(GECKO_SDK_BOARD_TARGET)
if(NOT GECKO_SDK_BOARD_TARGET)
    mltk_error("Must specify GECKO_SDK_BOARD_TARGET global property")
endif()


target_include_directories(${PROJECT_NAME} 
PUBLIC
    .
)

target_sources(${PROJECT_NAME}  
PRIVATE 
    sl_ml_audio_feature_generation.c
    sl_ml_audio_feature_generation_init.c 
    sl_ml_audio_feature_generation_config.cc
)

target_link_libraries(${PROJECT_NAME}  
PRIVATE
    ${GECKO_SDK_BOARD_TARGET}
    mltk::microfrontend
    mltk::cpputils
    mltk::tflite_micro
    mltk::tflite_model_parameters
)

mltk_get(MLTK_PLATFORM_IS_EMBEDDED)
if(MLTK_PLATFORM_IS_EMBEDDED)
    target_link_libraries(${PROJECT_NAME}  
    PRIVATE
        mltk::jlink_stream
        mltk::sl_mic
    )

    target_sources(${PROJECT_NAME}  
    PRIVATE 
        data_dumper_arm.cc
    )

else()
    target_link_libraries(${PROJECT_NAME}  
    PRIVATE
        mltk::libsoundio
    )

    target_sources(${PROJECT_NAME}  
    PRIVATE 
        data_dumper_pc.cc
    )
endif()