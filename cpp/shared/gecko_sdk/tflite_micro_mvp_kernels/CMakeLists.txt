project(mltk_tflite_micro_mvp_kernels
  VERSION 1.0.0
  DESCRIPTION "Silab's Gecko SDK MVP Tensorflow accelerated kernels"
)
export(PACKAGE ${PROJECT_NAME})
add_library(${PROJECT_NAME})
add_library(mltk::tflite_micro_mvp_kernels ALIAS ${PROJECT_NAME})

mltk_info("Including Gecko SDK MVP Tensorflow accelerated kernels" TAG mltk_tflite_micro_mvp_kernels)


# Find the common packages
find_package(mltk_gecko_sdk REQUIRED)
find_package(mltk_cpputils REQUIRED)
find_package(mltk_cmsis REQUIRED)



# Configure the additional libraries to link against
# based on the build platform
mltk_get(MLTK_PLATFORM_IS_EMBEDDED)

# If we're building for an embedded platform
if(MLTK_PLATFORM_IS_EMBEDDED)
  find_package(mltk_cmsis REQUIRED)
  target_sources(${PROJECT_NAME}
  PRIVATE
    compute/driver/mvp/src/sl_mvp.cc
    compute/driver/mvp/src/sl_mvp_power.cc
  )

  target_link_libraries(${PROJECT_NAME}
  PRIVATE
    mltk::tflite_micro_cmsis_kernels
  )

# Otherwise we're build for Window/Linux
else()
  mltk_info("Using MVP simulator" TAG mltk_tflite_micro_mvp_kernels)
  mltk_set(TFLITE_MICRO_SIMULATOR_ENABLED ON)

  # See if the GSDK MVP simulator source code package is
  # externally available. Ignore the error if not.
  find_package(mltk_sl_mvp_simulator QUIET)

  # If the MVP simulator was found externally
  # then link to the component
  if(mltk_sl_mvp_simulator_FOUND)
    target_compile_definitions(${PROJECT_NAME}
    PUBLIC
      SL_MVP_SIMULATOR_BUILD_FROM_SOURCE
    )
    target_link_libraries(${PROJECT_NAME}
    PUBLIC
      mltk::sl_mvp_simulator
    )

  # Otherwise we download and link to the pre-built static library
  else()
    execute_process(
      COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/simulator/download_simulator_lib.py
      RESULT_VARIABLE result
      OUTPUT_VARIABLE output
    )
    list(GET output 0 sl_mvp_simulator_lib_dir)
    if(result)
      if(output)
        list(GET output 1 _error_file_path)
        if(EXISTS "${_error_file_path}")
          file (STRINGS ${_error_file_path} _err_msg)
          string(REPLACE ";" "\n" _err_msg ${_err_msg})
        endif()
      endif()
      mltk_error("${_err_msg}\nFailed to download MVP simulator library, see: ${sl_mvp_simulator_lib_dir}/download.log\n\n" TAG mltk_tflite_micro_mvp_kernels)
    endif()

    string(REPLACE " " ";" _flag_list ${CMAKE_CXX_FLAGS})
    if("-fPIC" IN_LIST _flag_list)
      set(sl_mvp_simulator_lib_path "${sl_mvp_simulator_lib_dir}/sl_mvp_simulator.pic.a")
    else()
      set(sl_mvp_simulator_lib_path "${sl_mvp_simulator_lib_dir}/sl_mvp_simulator.a")
    endif()

    mltk_info("Using pre-built GSDK MVP simulator at ${sl_mvp_simulator_lib_path}" TAG mltk_tflite_micro_mvp_kernels)
    find_package(mltk_float16 REQUIRED)
    target_link_libraries(${PROJECT_NAME}
    PRIVATE
      ${sl_mvp_simulator_lib_path}
      mltk::float16
    )
    target_include_directories(${PROJECT_NAME}
    PUBLIC
      "${sl_mvp_simulator_lib_dir}"
    )

  endif()

endif()

target_sources(${PROJECT_NAME}
PRIVATE
  compute/driver/mvp/src/sl_mvp_program_area.cc
  compute/driver/mvp/src/sl_mvp_util.cc
  compute/nn/mvp/src/sl_mvp_ml_add.cc
  compute/nn/mvp/src/sl_mvp_ml_conv2d.cc
  compute/nn/mvp/src/sl_mvp_ml_depthwise_conv2d.cc
  compute/nn/mvp/src/sl_mvp_ml_fully_connected.cc
  compute/nn/mvp/src/sl_mvp_ml_mul.cc
  compute/nn/mvp/src/sl_mvp_ml_pooling.cc
  compute/nn/mvp/src/sl_mvp_ml_transpose_conv2d.cc
  compute/nn/mvp/src/sli_mvp_ml_depthwise_conv2d_opt.cc
  compute/nn/mvp/src/sli_mvp_ml_depthwise_conv2d.cc
  compute/math/mvp/src/sl_math_mvp_matrix_add.cc
  compute/math/mvp/src/sl_math_mvp_matrix_mult.cc
  compute/math/mvp/src/sl_math_mvp_matrix_scale.cc
  compute/math/mvp/src/sl_math_mvp_matrix_sub.cc
  compute/math/mvp/src/sl_math_mvp_matrix_transpose.cc
  compute/math/mvp/src/sl_math_mvp_matrix_vector_mult.cc
  compute/math/mvp/src/sl_math_mvp_vector_abs.cc
  compute/math/mvp/src/sl_math_mvp_vector_add.cc
  compute/math/mvp/src/sl_math_mvp_vector_clamp.cc
  compute/math/mvp/src/sl_math_mvp_vector_clip.cc
  compute/math/mvp/src/sl_math_mvp_vector_copy.cc
  compute/math/mvp/src/sl_math_mvp_vector_dot_product.cc
  compute/math/mvp/src/sl_math_mvp_vector_fill.cc
  compute/math/mvp/src/sl_math_mvp_vector_mult.cc
  compute/math/mvp/src/sl_math_mvp_vector_negate.cc
  compute/math/mvp/src/sl_math_mvp_vector_offset.cc
  compute/math/mvp/src/sl_math_mvp_vector_scale.cc
  compute/math/mvp/src/sl_math_mvp_vector_sub.cc
  compute/math/src/sl_math_matrix.cc
  kernels/add.cc
  kernels/conv.cc
  kernels/fully_connected.cc
  kernels/mul.cc
  kernels/pooling.cc
  kernels/depthwise_conv.cc
  kernels/transpose_conv.cc
  tflite_micro_mvp_accelerator.cc
)


mltk_append(TFLITE_MICRO_EXCLUDED_REF_KERNELS
  add
  conv
  depthwise_conv
  fully_connected
  mul
  pooling
  transpose_conv
)


# Specify the MVP accelerator
# before we find the tflite micro component.
# This will cause tflite micro to use the MVP kernels
# NOTE: We must find the tflite micro component AFTER
#       we specify the TFLITE_MICRO_EXCLUDED_REF_KERNELS variable
mltk_set(TFLITE_MICRO_ACCELERATOR mvp)
find_package(mltk_tflite_micro REQUIRED)


target_include_directories(${PROJECT_NAME}
PUBLIC
  compute/driver/mvp/inc
  compute/nn/mvp/inc
  compute/math/inc
  compute/math/mvp/inc
)

target_link_libraries(${PROJECT_NAME}
PUBLIC
  mltk::tflite_micro
  mltk::gecko_sdk::includes
  mltk::cmsis_nn
PRIVATE
  mltk::cpputils
)

mltk_get(MVP_USE_PROGRAMBUILDER)

if(MVP_USE_PROGRAMBUILDER)
  mltk_info("MVP_USE_PROGRAMBUILDER=ON, using program builder for MVP math lib" TAG mltk_tflite_micro_mvp_kernels)
  target_compile_definitions(${PROJECT_NAME}
  PRIVATE
    USE_MVP_PROGRAMBUILDER=1
  )
else()
  mltk_info("MVP_USE_PROGRAMBUILDER=OFF, using CMSIS for MVP math lib" TAG mltk_tflite_micro_mvp_kernels)
endif()