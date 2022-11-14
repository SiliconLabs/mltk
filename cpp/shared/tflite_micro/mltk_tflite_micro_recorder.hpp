#pragma once 

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "cpputils/helpers.hpp"
#include "msgpack.hpp"


namespace mltk
{

extern "C" 
{

DLL_EXPORT msgpack_context_t* get_layer_recording_context(bool force = false);
DLL_EXPORT void record_layer_conv_params(
  const tflite::ConvParams& params, 
  const int32_t* per_channel_output_multiplier, 
  const int32_t *per_channel_output_shift,
  int n_channels
);
DLL_EXPORT void record_layer_depthwise_conv_params(
  const tflite::DepthwiseParams& params, 
  const int32_t* per_channel_output_multiplier, 
  const int32_t *per_channel_output_shift,
  int n_channels
);
DLL_EXPORT void record_layer_fully_connected_params(
  const tflite::FullyConnectedParams& params
);
DLL_EXPORT void record_layer_pool_params(
  const tflite::PoolParams& params
);

} // extern "C" 


void reset_recorder();
bool get_recorded_data(const uint8_t** data_ptr, uint32_t* length_ptr);
void record_layer(int op_idx, const TfLiteContext& context, const TfLiteNode &node, bool record_input);


} // namespace mltk


#ifdef TFLITE_MICRO_RECORDER_ENABLED

#define TFLITE_MICRO_RESET_RECORDER() mltk::reset_recorder();
#define TFLITE_MICRO_RECORD_INPUTS(op_idx, context, node) mltk::record_layer(op_idx, *context, *node, true);
#define TFLITE_MICRO_RECORD_OUTPUTS(op_idx, context, node)  mltk::record_layer(op_idx, *context, *node, false);
#define TFLITE_MICRO_RECORD_CONV_PARAMS(op_params, per_channel_output_multiplier, per_channel_output_shift, n_channels) \
   mltk::record_layer_conv_params(op_params, per_channel_output_multiplier, per_channel_output_shift, n_channels)
#define TFLITE_MICRO_RECORD_DEPTHWISE_CONV_PARAMS(op_params, per_channel_output_multiplier, per_channel_output_shift, n_channels) \
   mltk::record_layer_depthwise_conv_params(op_params, per_channel_output_multiplier, per_channel_output_shift, n_channels)
#define TFLITE_MICRO_RECORD_FULLY_CONNECTED_PARAMS(op_params) mltk::record_layer_fully_connected_params(op_params)
#define TFLITE_MICRO_RECORD_POOL_PARAMS(op_params) mltk::record_layer_pool_params(op_params)

#else // TFLITE_MICRO_RECORDER_ENABLED

#define TFLITE_MICRO_RESET_RECORDER()
#define TFLITE_MICRO_RECORD_INPUTS(...)
#define TFLITE_MICRO_RECORD_OUTPUTS(...)
#define TFLITE_MICRO_RECORD_CONV_PARAMS(...)
#define TFLITE_MICRO_RECORD_DEPTHWISE_CONV_PARAMS(...)
#define TFLITE_MICRO_RECORD_FULLY_CONNECTED_PARAMS(...)
#define TFLITE_MICRO_RECORD_POOL_PARAMS(...)

#endif // TFLITE_MICRO_RECORDER_ENABLED

