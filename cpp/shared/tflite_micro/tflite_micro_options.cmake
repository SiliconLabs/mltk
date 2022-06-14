

mltk_define(TFLITE_MICRO_PROFILER_ENABLED
"Enable the MLTK profiling macros"
)


mltk_define(TFLITE_MICRO_ACCELERATOR_PROFILER_ENABLED
"Enable additional hardware accelerator profiling"
)

mltk_define(TFLITE_MICRO_RECORDER_ENABLED
"Enable the MLTK recording macros"
)

mltk_define(TFLITE_MICRO_ACCELERATOR
"The name of the accelerator to use for the optimized kernels"
)

mltk_define(TFLITE_MICRO_ACCELERATOR_RECORDER_ENABLED
"Enable recording accelerator data/instructions during inference"
)

mltk_define(TFLITE_MICRO_EXCLUDED_REF_KERNELS
"List of reference kernels to exclude from the build"
)

mltk_define(TFLITE_MICRO_SIMULATOR_ENABLED
"Enable the accelerator simulator"
)
