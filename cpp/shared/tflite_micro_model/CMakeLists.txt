project(mltk_tflite_micro_model
        VERSION 1.0.0
        DESCRIPTION "TF-Lite Micro model helper"
)
export (PACKAGE ${PROJECT_NAME})
add_library(${PROJECT_NAME})
add_library(mltk::tflite_micro_model ALIAS ${PROJECT_NAME})


find_package(mltk_tflite_model_parameters REQUIRED)
find_package(mltk_tflite_micro REQUIRED)
find_package(mltk_logging REQUIRED)
find_package(mltk_profiling REQUIRED)
find_package(mltk_gecko_sdk REQUIRED)


target_sources(${PROJECT_NAME}
PRIVATE 
    tflite_micro_model/tflite_micro_model.cc
    tflite_micro_model/tflite_micro_model_details.cc
    tflite_micro_model/tflite_micro_tensor.cc
)

target_include_directories(${PROJECT_NAME} 
PUBLIC 
    ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(${PROJECT_NAME}
PUBLIC
    mltk::tflite_model_parameters
    mltk::tflite_micro
    mltk::logging
    mltk::profiling
    mltk::gecko_sdk::includes
)
