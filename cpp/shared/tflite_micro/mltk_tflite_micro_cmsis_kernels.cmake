


###########################################################################
# CMSIS-Accelerated Kernels
#
mltk_get(MLTK_PLATFORM_IS_EMBEDDED)
if(MLTK_PLATFORM_IS_EMBEDDED)

find_package(mltk_cmsis REQUIRED)
add_library(mltk_tflite_micro_cmsis_kernels)
add_library(mltk::tflite_micro_cmsis_kernels ALIAS mltk_tflite_micro_cmsis_kernels)

set(tflm_cmsis_kernel_sources
  tensorflow/lite/micro/kernels/cmsis_nn/add.cc
  tensorflow/lite/micro/kernels/cmsis_nn/conv.cc
  tensorflow/lite/micro/kernels/cmsis_nn/depthwise_conv.cc
  tensorflow/lite/micro/kernels/cmsis_nn/fully_connected.cc
  tensorflow/lite/micro/kernels/cmsis_nn/mul.cc
  tensorflow/lite/micro/kernels/cmsis_nn/pooling.cc
  tensorflow/lite/micro/kernels/cmsis_nn/softmax.cc
  tensorflow/lite/micro/kernels/cmsis_nn/svdf.cc
)

# If an accelerator was previously included
# then exclude the CMSIS kernels that are accelerated
mltk_get(TFLITE_MICRO_EXCLUDED_REF_KERNELS)
if(TFLITE_MICRO_EXCLUDED_REF_KERNELS)
  mltk_info("Excluded CMSIS kernels: ${TFLITE_MICRO_EXCLUDED_REF_KERNELS}")
  foreach(pat ${TFLITE_MICRO_EXCLUDED_REF_KERNELS})
    list(FILTER tflm_cmsis_kernel_sources EXCLUDE REGEX ".*/${pat}\.cc")
  endforeach()
endif()

mltk_append(TFLITE_MICRO_EXCLUDED_REF_KERNELS
    add
    conv 
    depthwise_conv
    fully_connected
    mul
    pooling
    softmax
    svdf
)
list(TRANSFORM tflm_cmsis_kernel_sources PREPEND ${Tensorflow_SOURCE_BASE_DIR}/)
target_sources(mltk_tflite_micro_cmsis_kernels
PRIVATE 
  ${tflm_cmsis_kernel_sources}
)

target_link_libraries(mltk_tflite_micro_cmsis_kernels
PRIVATE 
  mltk::cmsis_nn
  mltk::tflite_micro
)

endif() # MLTK_PLATFORM_IS_EMBEDDED


