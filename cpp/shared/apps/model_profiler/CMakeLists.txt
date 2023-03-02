
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

mltk_get(MLTK_RUN_MODEL_FROM_RAM)
if(MLTK_RUN_MODEL_FROM_RAM)
    mltk_info("Loading .tflite flatbuffer into RAM" TAG mltk_model_profiler)
    target_compile_definitions(mltk_model_profiler
    PUBLIC
        MLTK_RUN_MODEL_FROM_RAM
    )
endif()



mltk_get(MLTK_RUNTIME_MEMORY_SIZE)
if(DEFINED MLTK_RUNTIME_MEMORY_SIZE)
    set(MODEL_PROFILER_RUNTIME_MEMORY_SIZE ${MLTK_RUNTIME_MEMORY_SIZE})
endif()
mltk_get(MODEL_PROFILER_RUNTIME_MEMORY_SIZE)
mltk_get(MODEL_PROFILER_RUNTIME_MEMORY_SECTION)
if(DEFINED MODEL_PROFILER_RUNTIME_MEMORY_SIZE)
    mltk_info("MODEL_PROFILER_RUNTIME_MEMORY_SIZE=${MODEL_PROFILER_RUNTIME_MEMORY_SIZE} bytes" TAG mltk_model_profiler)
    target_compile_definitions(mltk_model_profiler
    PRIVATE
        MODEL_PROFILER_RUNTIME_MEMORY_SIZE=${MODEL_PROFILER_RUNTIME_MEMORY_SIZE}
    )

    if(MODEL_PROFILER_RUNTIME_MEMORY_SECTION)
        mltk_info("MODEL_PROFILER_RUNTIME_MEMORY_SECTION=${MODEL_PROFILER_RUNTIME_MEMORY_SECTION}" TAG mltk_model_profiler)
        target_compile_definitions(mltk_model_profiler
        PRIVATE
            MODEL_PROFILER_RUNTIME_MEMORY_SECTION=\"${MODEL_PROFILER_RUNTIME_MEMORY_SECTION}\"
        )
    endif()
elseif(MODEL_PROFILER_RUNTIME_MEMORY_SECTION)
    mltk_warn("MODEL_PROFILER_RUNTIME_MEMORY_SECTION is not used if MODEL_PROFILER_RUNTIME_MEMORY_SIZE is not defined" TAG mltk_model_profiler)
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
    set(memory_section_arg MEMORY_SECTION ${MODEL_PROFILER_MODEL_MEMORY_SECTION})
    mltk_info("MODEL_PROFILER_MODEL_MEMORY_SECTION=${MODEL_PROFILER_MODEL_MEMORY_SECTION}" TAG mltk_model_profiler)
endif()
mltk_add_tflite_model(mltk_model_profiler "${MODEL_PROFILER_MODEL}" ${memory_section_arg})


mltk_get(MODEL_PROFILER_GENERATE_OP_RESOLVER_HEADER)
if(MODEL_PROFILER_GENERATE_OP_RESOLVER_HEADER)
    mltk_info("Generating OpResolver from ${MODEL_PROFILER_MODEL}" TAG mltk_model_profiler)
    mltk_add_tflite_model_op_resolver_header(mltk_model_profiler ${MODEL_PROFILER_MODEL})
endif()


# Generate the exe output files (if necessary for the build platform)
mltk_add_exe_targets(mltk_model_profiler)
