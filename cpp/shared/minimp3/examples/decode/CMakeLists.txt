

find_package(mltk_minimp3 REQUIRED)


# Define the hello_world executable
add_executable(mltk_minimp3_example
    main.cc
)  

# Generate the exe output files (if necessary for the build platform)
mltk_add_exe_targets(mltk_minimp3_example)

target_link_libraries(mltk_minimp3_example
PRIVATE 
    ${MLTK_PLATFORM}
    mltk::minimp3
)

