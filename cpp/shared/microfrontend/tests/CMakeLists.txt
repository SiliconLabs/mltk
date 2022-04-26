project(mltk_microfrontend_tests
        VERSION 1.0.0
        DESCRIPTION "MLTK Microfrontend Tests"
)
export(PACKAGE ${PROJECT_NAME})


add_executable(${PROJECT_NAME})


find_package(mltk_microfrontend REQUIRED)
find_package(mltk_gtest REQUIRED)

target_compile_features(${PROJECT_NAME}  PUBLIC cxx_constexpr cxx_std_17)

target_sources(${PROJECT_NAME}
PUBLIC 
    main.cc 
    fft_test.cc 
    filterbank_test.cc 
    frontend_test.cc 
    log_scale_test.cc 
    noise_reduction_test.cc 
    pcan_gain_control_test.cc 
    spectrogram_test.cc 
    window_test.cc
)

target_link_libraries( ${PROJECT_NAME}
PRIVATE 
    ${MLTK_PLATFORM}
    mltk::gtest
)

mltk_get(MLTK_PLATFORM_IS_EMBEDDED)
if(MLTK_PLATFORM_IS_EMBEDDED OR NOT HOST_OS_IS_WINDOWS)
    target_link_libraries(${PROJECT_NAME}
    PRIVATE 
        mltk::microfrontend
    )
else()
    target_link_libraries(${PROJECT_NAME}
    PRIVATE 
        microfrontend
    )
endif()

#####################################################
# Unit test

if(NOT MLTK_EXCLUDE_TESTS)
    add_test(mltk_microfrontend_tests ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/mltk_microfrontend_tests)
    set_tests_properties(mltk_microfrontend_tests
        PROPERTIES
        FAIL_REGULAR_EXPRESSION ".*FAILED.*")
endif()