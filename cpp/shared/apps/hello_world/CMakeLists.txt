

# Define the hello_world executable
add_executable(mltk_hello_world
    main.cc
)  

# Generate the exe output files (if necessary for the build platform)
mltk_add_exe_targets(mltk_hello_world)

target_link_libraries(mltk_hello_world
PRIVATE 
    ${MLTK_PLATFORM}
)

