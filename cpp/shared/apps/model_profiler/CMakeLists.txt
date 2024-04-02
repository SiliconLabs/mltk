
mltk_set(TFLITE_MICRO_PROFILER_ENABLED ON)


# Find the necessary packages
find_package(mltk_logging REQUIRED)
find_package(mltk_profiling REQUIRED)
find_package(mltk_tflite_micro_model REQUIRED)



#####################################################
# Define the model_profiler executable
add_executable(mltk_model_profiler)


target_sources(mltk_model_profiler
PRIVATE
    main.cc
)

target_link_libraries(mltk_model_profiler
PRIVATE
    mltk::tflite_micro_model
    ${MLTK_PLATFORM}
)

target_include_directories(mltk_model_profiler
PRIVATE
    .
)

mltk_get(MLTK_PLATFORM_IS_EMBEDDED)
if(NOT MLTK_PLATFORM_IS_EMBEDDED)
    find_package(mltk_cxxopts REQUIRED)
    target_link_libraries(mltk_model_profiler
    PRIVATE
        mltk::cxxopts
    )
    target_sources(mltk_model_profiler
    PRIVATE
        cli_opts.cc
    )

endif()



#####################################################
# Convert the model .tflite to a C array

# Use the default model if one wasn't provided on the command-line

mltk_get(MODEL_PROFILER_MODEL)
if(NOT MODEL_PROFILER_MODEL)
    # Support legacy variable name
    mltk_get(MODEL)
    if(MODEL)
        set(MODEL_PROFILER_MODEL ${MODEL})
    endif()
endif()

if(NOT MODEL_PROFILER_MODEL)
    set(MODEL_PROFILER_MODEL "image_classification")
endif()
mltk_info("MODEL_PROFILER_MODEL=${MODEL_PROFILER_MODEL}" TAG mltk_model_profiler)


mltk_get(MODEL_PROFILER_MODEL_MEMORY_SECTION)
if(MODEL_PROFILER_MODEL_MEMORY_SECTION)
    set(model_memory_section_arg MODEL_MEMORY_SECTION ${MODEL_PROFILER_MODEL_MEMORY_SECTION})
    mltk_info("MODEL_PROFILER_MODEL_MEMORY_SECTION=${MODEL_PROFILER_MODEL_MEMORY_SECTION}" TAG mltk_model_profiler)
endif()


mltk_get(MODEL_PROFILER_RUNTIME_MEMORY_SIZE)
mltk_get(MODEL_PROFILER_RUNTIME_MEMORY_SECTION)
if(DEFINED MODEL_PROFILER_RUNTIME_MEMORY_SIZE)
    mltk_info("MODEL_PROFILER_RUNTIME_MEMORY_SIZE=${MODEL_PROFILER_RUNTIME_MEMORY_SIZE} bytes" TAG mltk_model_profiler)
    set(runtime_memory_size_arg RUNTIME_MEMORY_SIZE ${MODEL_PROFILER_RUNTIME_MEMORY_SIZE})

    if(MODEL_PROFILER_RUNTIME_MEMORY_SECTION)
        mltk_info("MODEL_PROFILER_RUNTIME_MEMORY_SECTION=${MODEL_PROFILER_RUNTIME_MEMORY_SECTION}" TAG mltk_model_profiler)
        set(runtime_memory_section_arg RUNTIME_MEMORY_SECTION ${MODEL_PROFILER_RUNTIME_MEMORY_SECTION})
    endif()
elseif(MODEL_PROFILER_RUNTIME_MEMORY_SECTION)
    mltk_warn("MODEL_PROFILER_RUNTIME_MEMORY_SECTION is not used if MODEL_PROFILER_RUNTIME_MEMORY_SIZE is not defined" TAG mltk_model_profiler)
endif()


mltk_get(MODEL_PROFILER_GENERATE_OP_RESOLVER_HEADER)
if(MODEL_PROFILER_GENERATE_OP_RESOLVER_HEADER)
    mltk_info("Generating OpResolver from ${MODEL_PROFILER_MODEL}" TAG mltk_model_profiler)
    set(generate_op_resolver_arg GENERATE_OP_RESOLVER ON)
endif()

mltk_add_tflite_model(
    mltk_model_profiler "${MODEL_PROFILER_MODEL}" 
    GENERATE_MEMORY_LAYOUT ON
    ${model_memory_section_arg} 
    ${runtime_memory_section_arg}
    ${runtime_memory_size_arg}
    ${generate_op_resolver_arg}
)

# Generate the exe output files (if necessary for the build platform)
mltk_add_exe_targets(mltk_model_profiler)
