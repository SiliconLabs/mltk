#include "tensorflow/lite/schema/schema_generated.h"

#include "mltk_tflite_micro_recorder.hpp"
#include "mltk_tflite_micro_helper.hpp"


namespace mltk
{


static int padding_to_tflite_schema(tflite::PaddingType padding);


bool model_tensor_recorder_enabled = false;
static msgpack_context_t* _msgpack_context = nullptr;
static bool _root_array_finalized = false;
static bool _layer_started = false;


/*************************************************************************************************/
void reset_recorder()
{
  msgpack_buffered_writer_deinit(_msgpack_context, true);
  _msgpack_context = nullptr;
  _layer_started = false;
  _root_array_finalized = false;
}

/*************************************************************************************************/
bool start_recording()
{
  reset_recorder();
  if(msgpack_buffered_writer_init(&_msgpack_context, 32*1024) != 0)
  {
    return false;
  }
  msgpack_write_array_marker(_msgpack_context, -1);
  _root_array_finalized = false;
  _layer_started = false;

  return true;
}

/*************************************************************************************************/
bool get_recorded_data(const uint8_t** data_ptr, uint32_t* length_ptr)
{
  if(!_root_array_finalized)
  {
    _root_array_finalized = true;
    msgpack_finalize_dynamic(_msgpack_context);
  }

  return msgpack_buffered_writer_get_buffer(_msgpack_context, (uint8_t**)data_ptr, length_ptr) == 0;
}

/*************************************************************************************************/
void record_layer(
  int op_idx, 
  const TfLiteContext& context, 
  const TfLiteNode &node, 
  bool record_input
)
{
  if(_msgpack_context == nullptr)
  {
    if(!start_recording())
    {
      return;
    }
  }

  if(record_input)
  {
    _layer_started = true;
    msgpack_write_dict_marker(_msgpack_context, -1);

    if(model_tensor_recorder_enabled)
    {
      msgpack_write_dict_array(_msgpack_context, "inputs", node.inputs->size);

      for(int i = 0; i < node.inputs->size; ++i)
      {
        TfLiteTensor *tensor = nullptr;
        const int tensor_idx = node.inputs->data[i];
        if(tensor_idx >= 0)
        {
          tensor = context.GetTensor(&context, tensor_idx);
          msgpack_write_bin(_msgpack_context, tensor->data.raw, tensor->bytes);
        }
        else 
        {
          msgpack_write_nil(_msgpack_context);
        }
      }
    }
  }
  else 
  {
    if(model_tensor_recorder_enabled)
    {
      msgpack_write_dict_array(_msgpack_context, "outputs", node.outputs->size);

      for(int i = 0; i < node.outputs->size; ++i)
      {
        TfLiteTensor *tensor = nullptr;
        const int tensor_idx = node.outputs->data[i];
        if(tensor_idx >= 0)
        {
          tensor = context.GetTensor(&context, tensor_idx);
          msgpack_write_bin(_msgpack_context, tensor->data.raw, tensor->bytes);
        }
        else 
        {
          msgpack_write_nil(_msgpack_context);
        }
      }
    }

    msgpack_finalize_dynamic(_msgpack_context);
    _layer_started = false;
  }
}

/*************************************************************************************************/
#ifndef MLTK_DLL_IMPORT 
extern "C"  msgpack_context_t* get_layer_recording_context(bool force)
{
  if(!force && !model_tensor_recorder_enabled)
  {
    return nullptr;
  }

  return _layer_started ? _msgpack_context : nullptr;
}
#endif


/*************************************************************************************************/
#ifndef MLTK_DLL_IMPORT 
extern "C" void record_layer_conv_params(
  const tflite::ConvParams& params, 
  const int32_t* per_channel_output_multiplier, 
  const int32_t *per_channel_output_shift,
  int n_channels
)
{
  msgpack_context_t *msgpack = get_layer_recording_context();
  msgpack_write_dict_dict(msgpack, "params", -1);
  msgpack_write_dict_bin(msgpack, "per_channel_multiplier", per_channel_output_multiplier, sizeof(int32_t) * n_channels);
  msgpack_write_dict_bin(msgpack, "per_channel_shift", per_channel_output_shift, sizeof(int32_t) * n_channels);
  msgpack_write_dict_int(msgpack, "padding_type", padding_to_tflite_schema(params.padding_type));
  msgpack_write_dict_int(msgpack, "padding_width", params.padding_values.width);
  msgpack_write_dict_int(msgpack, "padding_height", params.padding_values.height);
  msgpack_write_dict_int(msgpack, "stride_width", params.stride_width);
  msgpack_write_dict_int(msgpack, "stride_height", params.stride_height);
  msgpack_write_dict_int(msgpack, "dilation_width_factor", params.dilation_width_factor);
  msgpack_write_dict_int(msgpack, "dilation_height_factor", params.dilation_height_factor);
  msgpack_write_dict_int(msgpack, "input_offset", params.input_offset);
  msgpack_write_dict_int(msgpack, "weights_offset", params.weights_offset);
  msgpack_write_dict_int(msgpack, "output_offset", params.output_offset);
  msgpack_write_dict_int(msgpack, "quantized_activation_min", params.quantized_activation_min);
  msgpack_write_dict_int(msgpack, "quantized_activation_max", params.quantized_activation_max);
  msgpack_finalize_dynamic(msgpack);
}
#endif // MLTK_DLL_IMPORT 


/*************************************************************************************************/
#ifndef MLTK_DLL_IMPORT
extern "C" void record_layer_depthwise_conv_params(
  const tflite::DepthwiseParams& params, 
  const int32_t* per_channel_output_multiplier, 
  const int32_t *per_channel_output_shift,
  int n_channels
)
{
  msgpack_context_t *msgpack = get_layer_recording_context();
  msgpack_write_dict_dict(msgpack, "params", -1);
  msgpack_write_dict_bin(msgpack, "per_channel_multiplier", per_channel_output_multiplier, sizeof(int32_t) * n_channels);
  msgpack_write_dict_bin(msgpack, "per_channel_shift", per_channel_output_shift, sizeof(int32_t) * n_channels);
  msgpack_write_dict_int(msgpack, "depth_multiplier", params.depth_multiplier);
  msgpack_write_dict_int(msgpack, "padding_type", padding_to_tflite_schema(params.padding_type));
  msgpack_write_dict_int(msgpack, "padding_width", params.padding_values.width);
  msgpack_write_dict_int(msgpack, "padding_height", params.padding_values.height);
  msgpack_write_dict_int(msgpack, "stride_width", params.stride_width);
  msgpack_write_dict_int(msgpack, "stride_height", params.stride_height);
  msgpack_write_dict_int(msgpack, "dilation_width_factor", params.dilation_width_factor);
  msgpack_write_dict_int(msgpack, "dilation_height_factor", params.dilation_height_factor);
  msgpack_write_dict_int(msgpack, "input_offset", params.input_offset);
  msgpack_write_dict_int(msgpack, "weights_offset", params.weights_offset);
  msgpack_write_dict_int(msgpack, "output_offset", params.output_offset);
  msgpack_write_dict_int(msgpack, "quantized_activation_min", params.quantized_activation_min);
  msgpack_write_dict_int(msgpack, "quantized_activation_max", params.quantized_activation_max);
  msgpack_finalize_dynamic(msgpack);
}
#endif // MLTK_DLL_IMPORT


/*************************************************************************************************/
#ifndef MLTK_DLL_IMPORT
extern "C"  void record_layer_fully_connected_params(
  const tflite::FullyConnectedParams& params
)
{
  msgpack_context_t *msgpack = get_layer_recording_context();
  msgpack_write_dict_dict(msgpack, "params", -1);
  msgpack_write_dict_int(msgpack, "input_offset", params.input_offset);
  msgpack_write_dict_int(msgpack, "weights_offset", params.weights_offset);
  msgpack_write_dict_int(msgpack, "output_offset", params.output_offset);
  msgpack_write_dict_int(msgpack, "output_multiplier", params.output_multiplier);
  msgpack_write_dict_int(msgpack, "output_shift", params.output_shift);
  msgpack_write_dict_int(msgpack, "quantized_activation_min", params.quantized_activation_min);
  msgpack_write_dict_int(msgpack, "quantized_activation_max", params.quantized_activation_max);
  msgpack_finalize_dynamic(msgpack);
}
#endif // MLTK_DLL_IMPORT


/*************************************************************************************************/
#ifndef MLTK_DLL_IMPORT
void record_layer_pool_params(
  const tflite::PoolParams& params
)
{
  msgpack_context_t *msgpack = get_layer_recording_context();
  msgpack_write_dict_dict(msgpack, "params", -1);
  msgpack_write_dict_int(msgpack, "padding_type", padding_to_tflite_schema(params.padding_type));
  msgpack_write_dict_int(msgpack, "padding_width", params.padding_values.width);
  msgpack_write_dict_int(msgpack, "padding_height", params.padding_values.height);
  msgpack_write_dict_int(msgpack, "stride_width", params.stride_width);
  msgpack_write_dict_int(msgpack, "stride_height", params.stride_height);
  msgpack_write_dict_int(msgpack, "filter_height", params.filter_height);
  msgpack_write_dict_int(msgpack, "filter_width", params.stride_height);
  msgpack_write_dict_int(msgpack, "quantized_activation_min", params.quantized_activation_min);
  msgpack_write_dict_int(msgpack, "quantized_activation_max", params.quantized_activation_max);
  msgpack_finalize_dynamic(msgpack);
}
#endif // MLTK_DLL_IMPORT



/*************************************************************************************************/
static int padding_to_tflite_schema(tflite::PaddingType padding)
{
  switch(padding)
  {
    case tflite::PaddingType::kSame:
      return tflite::Padding::Padding_SAME;
    case tflite::PaddingType::kValid:
      return tflite::Padding::Padding_VALID;
    default:
      return tflite::Padding::Padding_SAME;
  }
}


} // namespace mltk



