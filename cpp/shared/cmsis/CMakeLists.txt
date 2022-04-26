project(mltk_cmsis
        VERSION 1.0.0
        DESCRIPTION "ARM CMSIS Project")
export (PACKAGE ${PROJECT_NAME})
add_library(${PROJECT_NAME} INTERFACE)	 
add_library(mltk::cmsis ALIAS ${PROJECT_NAME})
add_library(mltk_cmsis_nn)	 
add_library(mltk::cmsis_nn ALIAS mltk_cmsis_nn)

include(${CMAKE_CURRENT_LIST_DIR}/cmsis_options.cmake)


# URL comes from:
# https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/tools/make/ext_libs/cmsis_download.sh
CPMAddPackage(
    NAME cmsis
    URL  http://github.com/ARM-software/CMSIS_5/archive/bff28575f0c96a4ee9008947fea2b018a69b4900.zip
    URL_HASH MD5=5412572a648d7103b311e8b7acd86a01
    DOWNLOAD_ONLY ON
    CACHE_VERSION v5.8.0
)

mltk_get(MLTK_PLATFORM_IS_EMBEDDED)
if(NOT MLTK_PLATFORM_IS_EMBEDDED)
    include("${CMAKE_CURRENT_LIST_DIR}/wrapper/wrapper_build.cmake")
    return()
endif()



set(compile_options "-mthumb -mno-thumb-interwork -mcpu=${CMSIS_CORE} -mfp16-format=ieee -I${cmsis_SOURCE_DIR}/CMSIS/Core/Include")

if(NOT CMSIS_CORE)
    mltk_error("Must define global variable CMSIS_CORE")
endif()

if(NOT CMSIS_FPU)
    set(compile_options "${compile_options} -mfloat-abi=softfp")
else()
    set(compile_options "${compile_options} -mfloat-abi=hard -mfpu=${CMSIS_FPU}")
endif()

if("${CMSIS_CORE}" STREQUAL "cortex-m4")
    set(compile_options "${compile_options}")
elseif("${CMSIS_CORE}" STREQUAL "cortex-m33")
    set(compile_options "${compile_options}")
elseif("${CMSIS_CORE}" STREQUAL "cortex-m3")
    set(compile_options "${compile_options}")
else()
    mltk_warn("CMSIS_CORE unsupported: ${CMSIS_CORE}")
endif()

# Append the CMSIS global compile options if they haven't already been added
if(NOT CMAKE_C_FLAGS MATCHES ".*${compile_options}.*")
    set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} ${compile_options}" CACHE INTERNAL "" FORCE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${compile_options}" CACHE INTERNAL "" FORCE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${compile_options}" CACHE INTERNAL "" FORCE)
endif()

target_include_directories(mltk_cmsis_nn
PUBLIC 
    ${cmsis_SOURCE_DIR}
    ${cmsis_SOURCE_DIR}/CMSIS/NN/Include
    ${cmsis_SOURCE_DIR}/CMSIS/DSP/Include
)
target_compile_options(mltk_cmsis_nn
PRIVATE 
    -Wno-strict-aliasing
)

set(sources
NN/Source/BasicMathFunctions/arm_elementwise_mul_s8.c
NN/Source/BasicMathFunctions/arm_elementwise_add_s8.c
NN/Source/FullyConnectedFunctions/arm_fully_connected_q7_opt.c
NN/Source/FullyConnectedFunctions/arm_fully_connected_s8.c
NN/Source/FullyConnectedFunctions/arm_fully_connected_q15_opt.c
NN/Source/FullyConnectedFunctions/arm_fully_connected_q15.c
NN/Source/FullyConnectedFunctions/arm_fully_connected_mat_q7_vec_q15.c
NN/Source/FullyConnectedFunctions/arm_fully_connected_q7.c
NN/Source/FullyConnectedFunctions/arm_fully_connected_mat_q7_vec_q15_opt.c
NN/Source/ConcatenationFunctions/arm_concatenation_s8_z.c
NN/Source/ConcatenationFunctions/arm_concatenation_s8_x.c
NN/Source/ConcatenationFunctions/arm_concatenation_s8_w.c
NN/Source/ConcatenationFunctions/arm_concatenation_s8_y.c
NN/Source/ConvolutionFunctions/arm_nn_depthwise_conv_s8_core.c
NN/Source/ConvolutionFunctions/arm_depthwise_conv_u8_basic_ver1.c
NN/Source/ConvolutionFunctions/arm_convolve_s8.c
NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8_fast.c
NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_fast.c
NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15_reordered.c
NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15.c
NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_RGB.c
NN/Source/ConvolutionFunctions/arm_nn_mat_mult_s8.c
NN/Source/ConvolutionFunctions/arm_convolve_1x1_HWC_q7_fast_nonsquare.c
NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_basic.c
NN/Source/ConvolutionFunctions/arm_depthwise_separable_conv_HWC_q7_nonsquare.c
NN/Source/ConvolutionFunctions/arm_convolve_HWC_q15_fast_nonsquare.c
NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8_opt.c
NN/Source/ConvolutionFunctions/arm_depthwise_conv_3x3_s8.c
NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.c
NN/Source/ConvolutionFunctions/arm_convolve_HWC_q15_basic.c
NN/Source/ConvolutionFunctions/arm_convolve_HWC_q15_fast.c
NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_fast_nonsquare.c
NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16_reordered.c
NN/Source/ConvolutionFunctions/arm_depthwise_separable_conv_HWC_q7.c
NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8.c
NN/Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.c
NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_basic_nonsquare.c
NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.c
NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s8.c
NN/Source/PoolingFunctions/arm_pool_q7_HWC.c
NN/Source/PoolingFunctions/arm_avgpool_s8.c
NN/Source/PoolingFunctions/arm_max_pool_s8.c
NN/Source/ActivationFunctions/arm_relu_q15.c
NN/Source/ActivationFunctions/arm_relu_q7.c
NN/Source/ActivationFunctions/arm_nn_activations_q7.c
NN/Source/ActivationFunctions/arm_relu6_s8.c
NN/Source/ActivationFunctions/arm_nn_activations_q15.c
NN/Source/NNSupportFunctions/arm_nn_add_q7.c
NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_padded_s8.c
NN/Source/NNSupportFunctions/arm_q7_to_q15_reordered_no_shift.c
NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s8.c
NN/Source/NNSupportFunctions/arm_nntables.c
NN/Source/NNSupportFunctions/arm_nn_accumulate_q7_to_q15.c
NN/Source/NNSupportFunctions/arm_nn_mult_q7.c
NN/Source/NNSupportFunctions/arm_nn_mult_q15.c
NN/Source/NNSupportFunctions/arm_q7_to_q15_reordered_with_offset.c
NN/Source/NNSupportFunctions/arm_q7_to_q15_with_offset.c
NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_4x_s8.c
NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c
NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.c
NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_svdf_s8.c
NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_1x_s8.c
NN/Source/NNSupportFunctions/arm_q7_to_q15_no_shift.c
NN/Source/SoftmaxFunctions/arm_softmax_s8.c
NN/Source/SoftmaxFunctions/arm_softmax_q15.c
NN/Source/SoftmaxFunctions/arm_softmax_u8.c
NN/Source/SoftmaxFunctions/arm_softmax_q7.c
NN/Source/SoftmaxFunctions/arm_softmax_with_batch_q7.c
NN/Source/ReshapeFunctions/arm_reshape_s8.c
NN/Source/SVDFunctions/arm_svdf_s8.c
)

list(TRANSFORM sources PREPEND ${cmsis_SOURCE_DIR}/CMSIS/)
target_sources(mltk_cmsis_nn
PRIVATE 
    ${sources}
)
target_compile_options(mltk_cmsis_nn
PRIVATE 
    -include stdint.h 
)
target_compile_definitions(mltk_cmsis_nn
PUBLIC
    CMSIS_NN
)

target_link_libraries(mltk_cmsis_nn
PUBLIC 
    mltk_cmsis
)
target_compile_definitions(mltk_cmsis_nn
PRIVATE 
    CMSIS_FORCE_BUILTIN_FUNCTIONS
)


add_library(mltk_cmsis_dsp_fft_q15)	 
add_library(mltk::cmsis_dsp_fft_q15 ALIAS mltk_cmsis_dsp_fft_q15)
target_include_directories(mltk_cmsis_dsp_fft_q15
PUBLIC 
    ${cmsis_SOURCE_DIR}
    ${cmsis_SOURCE_DIR}/CMSIS/DSP/Include
)

set(sources
DSP/Source/BasicMathFunctions/arm_shift_q15.c
DSP/Source/CommonTables/arm_common_tables.c
DSP/Source/CommonTables/arm_const_structs.c 
DSP/Source/TransformFunctions/arm_bitreversal.c
DSP/Source/TransformFunctions/arm_bitreversal2.c
DSP/Source/TransformFunctions/arm_cfft_init_q15.c
DSP/Source/TransformFunctions/arm_cfft_q15.c
DSP/Source/TransformFunctions/arm_cfft_radix2_init_q15.c 
DSP/Source/TransformFunctions/arm_cfft_radix2_q15.c 
DSP/Source/TransformFunctions/arm_cfft_radix4_init_q15.c 
DSP/Source/TransformFunctions/arm_cfft_radix4_q15.c 
DSP/Source/TransformFunctions/arm_rfft_init_q15.c 
DSP/Source/TransformFunctions/arm_rfft_q15.c 
)

list(TRANSFORM sources PREPEND ${cmsis_SOURCE_DIR}/CMSIS/)
target_sources(mltk_cmsis_dsp_fft_q15
PRIVATE 
    ${sources}
)
target_compile_options(mltk_cmsis_dsp_fft_q15
PRIVATE 
    -include stdint.h 
)


mltk_load_python()

# Ensure the downloaded library is patched 
add_custom_command(OUTPUT ${cmsis_SOURCE_DIR}/${PROJECT_NAME}_patch_complete.txt
  DEPENDS ${cmsis_SOURCE_DIR}/CMSIS/DSP/Include ${CMAKE_CURRENT_LIST_DIR}/patch_cmsis.py
  COMMAND ${PYTHON_EXECUTABLE} ${MLTK_CPP_UTILS_DIR}/libpatcher.py -i ${cmsis_SOURCE_DIR}/CMSIS/DSP/Include -p ${CMAKE_CURRENT_LIST_DIR}/patch_cmsis.py -o ${cmsis_SOURCE_DIR}/${PROJECT_NAME}_patch_complete.txt
)
add_custom_target(${PROJECT_NAME}_apply_patch DEPENDS ${cmsis_SOURCE_DIR}/${PROJECT_NAME}_patch_complete.txt)
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_apply_patch)