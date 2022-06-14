
# This app currently only supports embedded platforms
if(NOT MLTK_PLATFORM_IS_EMBEDDED)
    return()
endif()


# Find the necessary packages
find_package(mltk_logging REQUIRED)
find_package(mltk_profiling REQUIRED)
find_package(mltk_tflite_micro_model REQUIRED)
find_package(mltk_fingerprint_reader REQUIRED)



#####################################################
# Define the fingerprint_authenticator executable
add_executable(mltk_fingerprint_authenticator)  


target_sources(mltk_fingerprint_authenticator
PRIVATE 
    main.cc
    app_controller.cc 
    fingerprint_authenticator.cc
    data_preprocessor.cc 
    fingerprint_vault.c
)

target_link_libraries(mltk_fingerprint_authenticator
PRIVATE 
    mltk::tflite_micro_model
    mltk::fingerprint_reader
    ${MLTK_PLATFORM}
)


target_include_directories(mltk_fingerprint_authenticator
PRIVATE 
    .
)


#####################################################
# Convert the model .tflite to a C header file

# Use the default model if one wasn't provided on the command-line
mltk_get(FINGERPRINT_AUTHENTICATOR_MODEL)
if(NOT FINGERPRINT_AUTHENTICATOR_MODEL)
    set(FINGERPRINT_AUTHENTICATOR_MODEL "fingerprint_signature_generator")
endif()
mltk_info("FINGERPRINT_AUTHENTICATOR_MODEL=${FINGERPRINT_AUTHENTICATOR_MODEL}")

mltk_add_tflite_model(mltk_fingerprint_authenticator ${FINGERPRINT_AUTHENTICATOR_MODEL})


# Generate the exe output files (if necessary for the build platform)
mltk_add_exe_targets(mltk_fingerprint_authenticator)
