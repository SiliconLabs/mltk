
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

# There's also:
# mltk_info()
# mltk_warn()
# mltk_error()
# These levels are controlled by the CMake variable:
# MLTK_CMAKE_LOG_LEVEL, e.g.:
# mltk_set(MLTK_CMAKE_LOG_LEVEL debug)
mltk_debug("Example CMake debug msg" TAG mltk_hello_world)