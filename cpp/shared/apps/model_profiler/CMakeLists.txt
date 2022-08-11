

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


mltk_get(MLTK_RUN_MODEL_FROM_RAM)
if(MLTK_RUN_MODEL_FROM_RAM)
    mltk_info("Loading .tflite flatbuffer into RAM")
    target_compile_definitions(mltk_model_profiler
    PUBLIC 
        MLTK_RUN_MODEL_FROM_RAM
    )
endif()

mltk_get(MLTK_RUNTIME_MEMORY_SIZE)
if(DEFINED MLTK_RUNTIME_MEMORY_SIZE)
    mltk_info("Forcing run-time memory size to ${MLTK_RUNTIME_MEMORY_SIZE} bytes")
    target_compile_definitions(mltk_model_profiler
    PUBLIC 
        MLTK_RUNTIME_MEMORY_SIZE=${MLTK_RUNTIME_MEMORY_SIZE}
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
mltk_info("MODEL_PROFILER_MODEL=${MODEL_PROFILER_MODEL}")


mltk_add_tflite_model(mltk_model_profiler ${MODEL_PROFILER_MODEL})

# Generate the exe output files (if necessary for the build platform)
mltk_add_exe_targets(mltk_model_profiler)
