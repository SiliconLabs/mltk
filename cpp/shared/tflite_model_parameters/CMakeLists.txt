project(mltk_tflite_model_parameters
        VERSION 1.0.0
        DESCRIPTION "tflite model parameters"
)
export (PACKAGE ${PROJECT_NAME})


find_package(mltk_flatbuffers REQUIRED)
find_package(mltk_tflite_micro REQUIRED)

add_library(${PROJECT_NAME})
add_library(mltk::tflite_model_parameters ALIAS ${PROJECT_NAME})

target_sources(${PROJECT_NAME}
PRIVATE 
    tflite_model_parameters/tflite_model_parameters.cc
)

target_include_directories(${PROJECT_NAME} 
PUBLIC 
    ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(${PROJECT_NAME}
PUBLIC
    mltk::flatbuffers
    mltk::tflite_micro
)
