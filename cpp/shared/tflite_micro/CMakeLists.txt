project(mltk_tflite_micro
        VERSION 1.0.0
        DESCRIPTION "Wrapper project for Google Tensorflow-Lite Micro"
        HOMEPAGE_URL "https://github.com/tensorflow/tflite-micro"
)
export(PACKAGE ${PROJECT_NAME})
add_library(${PROJECT_NAME}) 
add_library(mltk::tflite_micro ALIAS ${PROJECT_NAME})

include(${CMAKE_CURRENT_LIST_DIR}/tflite_micro_options.cmake)


mltk_info("Processing Tensorflow-Lite Micro repository (this may take awhile) ...")

# NOTE: When updating the TFLM version, be sure to also update the Simplicity Studio files:
# <mltk root>/cpp/shared/gecko_sdk/simplicity_studio/mltk_tflite_micro.slcc
# <mltk root>/cpp/shared/gecko_sdk/simplicity_studio/mltk_tflite_micro_cmsis_kernels.slcc
# <mltk root>/cpp/shared/gecko_sdk/simplicity_studio/mltk_tflite_micro_reference_kernels.slcc
# Also be sure to update the TFLITE_MICRO_VERSION_STR define in the mltk_tflite_micro.slcc
# Also update the <mltk root>/README.md with the version change
set(_tflm_cache_version "nov8_2022")
CPMAddPackage(
NAME Tensorflow
  GITHUB_REPOSITORY tensorflow/tflite-micro
  GIT_TAG 370d7bbc85f72bdb6004a8bfb7e778c271864ee2 # latest master as of November 8th, 2022
  DOWNLOAD_ONLY ON
  CACHE_SUBDIR tflite_micro
  CACHE_VERSION ${_tflm_cache_version}
)

set(Tensorflow_SOURCE_DIR ${Tensorflow_SOURCE_DIR} CACHE INTERNAL "")
set(Tensorflow_SOURCE_BASE_DIR "tensorflow/${_tflm_cache_version}")

mltk_git_version(${Tensorflow_SOURCE_DIR} TFLITE_MICRO_VERSION)
mltk_info("TFLITE_MICRO_VERSION: ${TFLITE_MICRO_VERSION}")


find_package(mltk_gemmlowp REQUIRED)
find_package(mltk_ruy REQUIRED)
find_package(mltk_flatbuffers REQUIRED)
find_package(mltk_profiling REQUIRED)
find_package(mltk_logging REQUIRED)
find_package(mltk_cpputils REQUIRED)
find_package(mltk_float16 REQUIRED)
find_package(mltk_cmsis REQUIRED)
find_package(mltk_msgpack REQUIRED)


####################################################
# Determine whcih kernels to use
mltk_get(TFLITE_MICRO_ACCELERATOR)
mltk_get(MLTK_PLATFORM_IS_EMBEDDED)

if(TFLITE_MICRO_ACCELERATOR)
    mltk_info("Using ${TFLITE_MICRO_ACCELERATOR} kernels")
    find_package(mltk_tflite_micro_${TFLITE_MICRO_ACCELERATOR}_kernels REQUIRED)
    set(tflm_kernels_target mltk_tflite_micro_${TFLITE_MICRO_ACCELERATOR}_kernels)
elseif(MLTK_PLATFORM_IS_EMBEDDED)
    mltk_info("Using CMSIS kernels")
    set(tflm_kernels_target mltk_tflite_micro_cmsis_kernels)
else()
    mltk_info("Using Reference kernels")
    set(tflm_kernels_target)
endif()

# Include the CMSIS kernels AFTER the accelerator has been (potentially) included
# This way the accelerator can exclude CMSIS kernels if necessary
include(${CMAKE_CURRENT_LIST_DIR}/mltk_tflite_micro_cmsis_kernels.cmake)




###########################################################################
# Tensorflow-Lite Micro base library
#


set(tflm_sources 
tensorflow/lite/c/common.cc
tensorflow/lite/core/api/flatbuffer_conversions.cc
tensorflow/lite/core/api/op_resolver.cc
tensorflow/lite/core/api/tensor_utils.cc
tensorflow/lite/core/api/error_reporter.cc
tensorflow/lite/schema/schema_utils.cc
tensorflow/lite/kernels/internal/quantization_util.cc
tensorflow/lite/kernels/internal/portable_tensor_utils.cc
tensorflow/lite/kernels/internal/tensor_utils.cc 
tensorflow/lite/kernels/internal/reference/portable_tensor_utils.cc
tensorflow/lite/kernels/kernel_util.cc 
tensorflow/lite/micro/arena_allocator/non_persistent_arena_buffer_allocator.cc
tensorflow/lite/micro/arena_allocator/persistent_arena_buffer_allocator.cc
tensorflow/lite/micro/arena_allocator/recording_single_arena_buffer_allocator.cc
tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.cc
tensorflow/lite/micro/all_ops_resolver.cc 
tensorflow/lite/micro/debug_log.cc 
tensorflow/lite/micro/fake_micro_context.cc
tensorflow/lite/micro/flatbuffer_utils.cc
tensorflow/lite/micro/memory_helpers.cc
tensorflow/lite/micro/micro_allocation_info.cc
tensorflow/lite/micro/micro_allocator.cc
tensorflow/lite/micro/micro_context.cc
tensorflow/lite/micro/micro_error_reporter.cc
#tensorflow/lite/micro/micro_graph.cc # This is overridden to enable recording and profiling
tensorflow/lite/micro/micro_interpreter.cc
#tensorflow/lite/micro/micro_log.cc  # This is overridden
tensorflow/lite/micro/micro_profiler.cc
tensorflow/lite/micro/micro_resource_variable.cc
tensorflow/lite/micro/micro_string.cc
tensorflow/lite/micro/micro_time.cc
tensorflow/lite/micro/micro_utils.cc
tensorflow/lite/micro/mock_micro_graph.cc
tensorflow/lite/micro/test_helpers.cc
tensorflow/lite/micro/test_helper_custom_ops.cc
tensorflow/lite/micro/kernels/kernel_runner.cc 
tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc
tensorflow/lite/micro/memory_planner/linear_memory_planner.cc
tensorflow/lite/micro/kernels/kernel_util.cc
tensorflow/lite/micro/kernels/micro_tensor_utils.cc
tensorflow/lite/micro/kernels/activations.cc
tensorflow/lite/micro/kernels/activations_common.cc
tensorflow/lite/micro/kernels/add.cc
tensorflow/lite/micro/kernels/add_common.cc
tensorflow/lite/micro/kernels/add_n.cc
tensorflow/lite/micro/kernels/arg_min_max.cc
tensorflow/lite/micro/kernels/assign_variable.cc
tensorflow/lite/micro/kernels/batch_to_space_nd.cc
tensorflow/lite/micro/kernels/broadcast_args.cc
tensorflow/lite/micro/kernels/broadcast_to.cc
tensorflow/lite/micro/kernels/call_once.cc
tensorflow/lite/micro/kernels/cast.cc
tensorflow/lite/micro/kernels/ceil.cc
tensorflow/lite/micro/kernels/circular_buffer.cc
tensorflow/lite/micro/kernels/circular_buffer_common.cc
tensorflow/lite/micro/kernels/comparisons.cc
#tensorflow/lite/micro/kernels/concatenation.cc # This is overriden to allow for concatenating more than 12 kernels
tensorflow/lite/micro/kernels/conv_common.cc
#tensorflow/lite/micro/kernels/conv.cc # This is overridden to account for the CMSIS scratch buffers for arena size calculation
tensorflow/lite/micro/kernels/cumsum.cc
tensorflow/lite/micro/kernels/depth_to_space.cc
tensorflow/lite/micro/kernels/depthwise_conv_common.cc
#tensorflow/lite/micro/kernels/depthwise_conv.cc # This is overridden to account for the CMSIS scratch buffers for arena size calculation
tensorflow/lite/micro/kernels/dequantize.cc
tensorflow/lite/micro/kernels/dequantize_common.cc
tensorflow/lite/micro/kernels/detection_postprocess.cc
tensorflow/lite/micro/kernels/div.cc
tensorflow/lite/micro/kernels/elementwise.cc
tensorflow/lite/micro/kernels/elu.cc
tensorflow/lite/micro/kernels/ethosu.cc
tensorflow/lite/micro/kernels/exp.cc
tensorflow/lite/micro/kernels/expand_dims.cc
tensorflow/lite/micro/kernels/fill.cc
tensorflow/lite/micro/kernels/floor_div.cc
tensorflow/lite/micro/kernels/floor_mod.cc
tensorflow/lite/micro/kernels/floor.cc
tensorflow/lite/micro/kernels/fully_connected_common.cc
#tensorflow/lite/micro/kernels/fully_connected.cc  # This is overridden to account for the CMSIS scratch buffers for arena size calculation
tensorflow/lite/micro/kernels/gather_nd.cc
tensorflow/lite/micro/kernels/gather.cc
tensorflow/lite/micro/kernels/hard_swish_common.cc
tensorflow/lite/micro/kernels/hard_swish.cc
tensorflow/lite/micro/kernels/if.cc
tensorflow/lite/micro/kernels/l2_pool_2d.cc
tensorflow/lite/micro/kernels/l2norm.cc
tensorflow/lite/micro/kernels/leaky_relu.cc
tensorflow/lite/micro/kernels/leaky_relu_common.cc
tensorflow/lite/micro/kernels/log_softmax.cc
tensorflow/lite/micro/kernels/logical_common.cc
tensorflow/lite/micro/kernels/logical.cc
tensorflow/lite/micro/kernels/logistic_common.cc
tensorflow/lite/micro/kernels/logistic.cc
tensorflow/lite/micro/kernels/lstm_eval.cc
tensorflow/lite/micro/kernels/maximum_minimum.cc
tensorflow/lite/micro/kernels/mirror_pad.cc
tensorflow/lite/micro/kernels/mul.cc
tensorflow/lite/micro/kernels/mul_common.cc
tensorflow/lite/micro/kernels/neg.cc
tensorflow/lite/micro/kernels/pad.cc
tensorflow/lite/micro/kernels/pack.cc
tensorflow/lite/micro/kernels/pooling_common.cc
#tensorflow/lite/micro/kernels/pooling.cc # This is overridden to account for the CMSIS scratch buffers for arena size calculation
tensorflow/lite/micro/kernels/prelu.cc
tensorflow/lite/micro/kernels/prelu_common.cc
tensorflow/lite/micro/kernels/quantize.cc
tensorflow/lite/micro/kernels/quantize_common.cc
tensorflow/lite/micro/kernels/read_variable.cc
tensorflow/lite/micro/kernels/reduce_common.cc
tensorflow/lite/micro/kernels/reduce.cc
tensorflow/lite/micro/kernels/reshape.cc
tensorflow/lite/micro/kernels/resize_bilinear.cc
tensorflow/lite/micro/kernels/resize_nearest_neighbor.cc
tensorflow/lite/micro/kernels/round.cc
tensorflow/lite/micro/kernels/select.cc
tensorflow/lite/micro/kernels/shape.cc
tensorflow/lite/micro/kernels/slice.cc
tensorflow/lite/micro/kernels/softmax_common.cc
tensorflow/lite/micro/kernels/softmax.cc
tensorflow/lite/micro/kernels/space_to_batch_nd.cc
tensorflow/lite/micro/kernels/space_to_depth.cc
tensorflow/lite/micro/kernels/split.cc
tensorflow/lite/micro/kernels/split_v.cc
tensorflow/lite/micro/kernels/squared_difference.cc
tensorflow/lite/micro/kernels/squeeze.cc
tensorflow/lite/micro/kernels/strided_slice.cc
tensorflow/lite/micro/kernels/sub.cc
tensorflow/lite/micro/kernels/sub_common.cc
tensorflow/lite/micro/kernels/svdf_common.cc
tensorflow/lite/micro/kernels/svdf.cc
tensorflow/lite/micro/kernels/tanh.cc
tensorflow/lite/micro/kernels/unpack.cc
tensorflow/lite/micro/kernels/transpose_conv.cc
tensorflow/lite/micro/kernels/transpose.cc
tensorflow/lite/micro/kernels/unidirectional_sequence_lstm.cc
tensorflow/lite/micro/kernels/unpack.cc
tensorflow/lite/micro/kernels/var_handle.cc
tensorflow/lite/micro/kernels/while.cc
tensorflow/lite/micro/kernels/zeros_like.cc
)

set(tflm_overidden_kernels 
  kernels/concatenation.cc
  kernels/conv.cc
  kernels/depthwise_conv.cc
  kernels/fully_connected.cc
  kernels/pooling.cc
)

# Exclude any reference kernels
# This is used by the CMSIS and accelerator kernels that have their own
# optimized implementations
mltk_get(TFLITE_MICRO_EXCLUDED_REF_KERNELS)
if(TFLITE_MICRO_EXCLUDED_REF_KERNELS)
  mltk_info("Excluded TF-Lite Micro reference kernels: ${TFLITE_MICRO_EXCLUDED_REF_KERNELS}")
  foreach(pat ${TFLITE_MICRO_EXCLUDED_REF_KERNELS})
    list(FILTER tflm_sources EXCLUDE REGEX ".*/${pat}\.cc")
    list(FILTER tflm_overidden_kernels EXCLUDE REGEX ".*/${pat}\.cc")
  endforeach()
endif()
 


list(TRANSFORM tflm_sources PREPEND ${Tensorflow_SOURCE_BASE_DIR}/)
target_sources(${PROJECT_NAME}
PRIVATE 
    ${tflm_sources}
    ${tflm_overidden_kernels}
    micro_graph.cc
    micro_log.cc
    mltk_calculate_op_metrics.cc
    mltk_tflite_micro_helper.cc
    mltk_tflite_micro_internal.cc
    mltk_tflite_micro_recorder.cc
)


target_compile_definitions(${PROJECT_NAME} 
PUBLIC 
  TF_LITE_USE_GLOBAL_ROUND
  TFLITE_SINGLE_ROUNDING
  TF_LITE_STATIC_MEMORY
PRIVATE
  TFLITE_MICRO_VERSION_STR="${TFLITE_MICRO_VERSION}"
)

if(TFLITE_MICRO_ACCELERATOR)
  mltk_info("TFLITE_MICRO_ACCELERATOR=${TFLITE_MICRO_ACCELERATOR}")
  target_compile_definitions(${PROJECT_NAME} 
  PUBLIC 
    TFLITE_MICRO_ACCELERATOR="${TFLITE_MICRO_ACCELERATOR}"
  )
endif()

mltk_get(TFLITE_MICRO_ACCELERATOR_PROFILER_ENABLED)
if(TFLITE_MICRO_ACCELERATOR_PROFILER_ENABLED)
  mltk_info("TFLITE_MICRO_ACCELERATOR_PROFILER_ENABLED=ON, Using accelerator profiling")
  target_compile_definitions(${PROJECT_NAME} 
  PUBLIC 
    TFLITE_MICRO_ACCELERATOR_PROFILER_ENABLED
  )
endif()


mltk_get(TFLITE_MICRO_PROFILER_ENABLED)
if(TFLITE_MICRO_PROFILER_ENABLED OR TFLITE_MICRO_ACCELERATOR_PROFILER_ENABLED)
  mltk_info("TFLITE_MICRO_PROFILER_ENABLED=ON, Enabling TFLM profiler")
  target_compile_definitions(${PROJECT_NAME} 
  PUBLIC 
    TFLITE_MICRO_PROFILER_ENABLED
  )
endif()

mltk_get(TFLITE_MICRO_RECORDER_ENABLED)
if(TFLITE_MICRO_RECORDER_ENABLED)
  mltk_info("TFLITE_MICRO_RECORDER_ENABLED=ON, Enabling TFLM recorder")
  target_compile_definitions(${PROJECT_NAME} 
  PUBLIC 
    TFLITE_MICRO_RECORDER_ENABLED
  )
endif()

mltk_get(TFLITE_MICRO_ACCELERATOR_RECORDER_ENABLED)
if(TFLITE_MICRO_ACCELERATOR_RECORDER_ENABLED)
  mltk_info("TFLITE_MICRO_ACCELERATOR_RECORDER_ENABLED=ON, Using accelerator instruction recorder")
  target_compile_definitions(${PROJECT_NAME} 
  PUBLIC 
    TFLITE_MICRO_ACCELERATOR_RECORDER_ENABLED
  )
endif()


mltk_get(TFLITE_MICRO_SIMULATOR_ENABLED)
if(TFLITE_MICRO_SIMULATOR_ENABLED)
  mltk_info("TFLITE_MICRO_SIMULATOR_ENABLED=ON, Enabling TFLM simulator")
  target_compile_definitions(${PROJECT_NAME} 
  PUBLIC 
    TFLITE_MICRO_SIMULATOR_ENABLED
  )
endif()

mltk_get(TFLITE_MICRO_OVERRIDE_QUANTIZED_MULTIPLIER_API)
if(DEFINED TFLITE_MICRO_OVERRIDE_QUANTIZED_MULTIPLIER_API)
  mltk_warn("TFLITE_MICRO_OVERRIDE_QUANTIZED_MULTIPLIER_API is deprecated as this option is now always enabled")
endif()

mltk_get(MLTK_LOG_LEVEL)
if(DEFINED MLTK_LOG_LEVEL)
  mltk_info("MLTK_LOG_LEVEL=${MLTK_LOG_LEVEL}")
  target_compile_definitions(${PROJECT_NAME} 
  PUBLIC 
    MLTK_LOG_LEVEL=${MLTK_LOG_LEVEL}
  )
endif()


target_link_libraries(${PROJECT_NAME} 
PUBLIC
  mltk::platform::common
  mltk::gemmlowp 
  mltk::flatbuffers
  mltk::ruy
  mltk::profiling
  mltk::logging
  mltk::float16
  mltk::msgpack
PRIVATE 
  ${tflm_kernels_target}
  mltk::cmsis_nn
  mltk::cpputils
)

target_include_directories(${PROJECT_NAME} 
PUBLIC 
    ${Tensorflow_SOURCE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}
)	 

target_compile_features(${PROJECT_NAME}  
PUBLIC 
  cxx_constexpr 
  cxx_std_11
)

target_compile_options(${PROJECT_NAME}  
PUBLIC 
  -Wno-sign-compare
  -Wno-unused-but-set-variable
  -Wno-comment
  -Wno-type-limits
  -Wno-strict-aliasing
  -Wno-missing-field-initializers
  -Wno-deprecated-declarations
  -Wno-maybe-uninitialized
  -Wno-unused-variable
  -Wno-unused-value
  -Wno-psabi
)


mltk_load_python()

# Ensure the downloaded library is patched 
add_custom_command(OUTPUT ${Tensorflow_SOURCE_DIR}/${PROJECT_NAME}_patch_complete.txt
  DEPENDS ${Tensorflow_SOURCE_DIR}/tensorflow/lite/micro ${CMAKE_CURRENT_LIST_DIR}/patch_tensorflow.py
  COMMAND ${PYTHON_EXECUTABLE} ${MLTK_CPP_UTILS_DIR}/libpatcher.py -i ${Tensorflow_SOURCE_DIR}/tensorflow/lite -p ${CMAKE_CURRENT_LIST_DIR}/patch_tensorflow.py -o ${Tensorflow_SOURCE_DIR}/${PROJECT_NAME}_patch_complete.txt
)
add_custom_target(${PROJECT_NAME}_apply_patch DEPENDS ${Tensorflow_SOURCE_DIR}/${PROJECT_NAME}_patch_complete.txt)
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_apply_patch)
