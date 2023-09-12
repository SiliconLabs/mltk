set(IMAGE_CLASSIFIER_SUPPORTED_PLATFORMS brd2204 brd2601 brd4166 brd4186)
if(NOT ${MLTK_PLATFORM_NAME} IN_LIST IMAGE_CLASSIFIER_SUPPORTED_PLATFORMS)
    mltk_info(
        "Image Classifier app NOT currently supported by platform: ${MLTK_PLATFORM_NAME}"
        TAG mltk_image_classifier
    )
    return()
endif()


# Find the necessary packages
find_package(mltk_logging REQUIRED)
find_package(mltk_profiling REQUIRED)
find_package(mltk_tflite_micro_model REQUIRED)
find_package(mltk_arducam REQUIRED)



#####################################################
# Define the image_classifier executable
add_executable(mltk_image_classifier)


target_sources(mltk_image_classifier
PRIVATE
    main.cc
    app.cc
    image_classifier.cc
    recognize_commands.cc
)

target_link_libraries(mltk_image_classifier
PRIVATE
    mltk::tflite_micro_model
    mltk::arducam
    ${MLTK_PLATFORM}
)


target_include_directories(mltk_image_classifier
PRIVATE
    .
)


#####################################################
# Convert the model .tflite to a C  array

# Use the default model if one wasn't provided on the command-line
mltk_get(IMAGE_CLASSIFIER_MODEL)
if(NOT IMAGE_CLASSIFIER_MODEL)
    set(IMAGE_CLASSIFIER_MODEL "rock_paper_scissors")
endif()
mltk_info("IMAGE_CLASSIFIER_MODEL=${IMAGE_CLASSIFIER_MODEL}" TAG mltk_image_classifier)

mltk_add_tflite_model(mltk_image_classifier ${IMAGE_CLASSIFIER_MODEL})


# Generate the exe output files (if necessary for the build platform)
mltk_add_exe_targets(mltk_image_classifier)
