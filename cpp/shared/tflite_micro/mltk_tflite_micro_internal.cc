#include <cstdarg>

#include "mltk_tflite_micro_internal.hpp"
#include "mltk_tflite_micro_recorded_data.hpp"


namespace mltk
{

profiling::Profiler *_inference_profiler = nullptr;
profiling::Profiler **_kernel_profilers = nullptr;
int _current_kernel_index = -1;
int _current_kernel_op_code = -1;
bool _issued_unsupported_msg = false;
void (*_processing_callback)(void*) = nullptr;
void* _processing_callback_arg = nullptr;;



/*************************************************************************************************/
void allocate_profilers(int subgraph_index, int op_count)
{
  if(subgraph_index > 0 || !model_profiler_enabled)
  {
    return;
  }
  if(!profiling::register_profiler("Inference", _inference_profiler))
  {
    return;
  }
  _inference_profiler->flags(profiling::Flag::ReportTotalChildrenCycles|profiling::Flag::ReportsFreeRunningCpuCycles);
  _kernel_profilers = static_cast<profiling::Profiler**>(malloc(sizeof(profiling::Profiler*)*op_count));
  if(_kernel_profilers == nullptr)
  {
    return;
  }
}

/*************************************************************************************************/
void free_profilers()
{
    if(_inference_profiler != nullptr)
    {
        // Unregister the inference profiler and all its children profilers
        profiling::unregister(_inference_profiler);
        _inference_profiler = nullptr;
    }
    if(_kernel_profilers != nullptr)
    {
        free(_kernel_profilers);
        _kernel_profilers = nullptr;
    }
}

/*************************************************************************************************/
void register_profiler(
  int subgraph_idx, 
  int op_idx, 
  tflite::BuiltinOperator op_type,
  const TfLiteContext* context,
  const tflite::NodeAndRegistration& node_and_registration 
)
{
  if(subgraph_idx > 0 || _kernel_profilers == nullptr)
  {
    return;
  }
  profiling::Profiler* profiler;
  profiling::register_profiler(op_to_str(op_idx, op_type), profiler, _inference_profiler);
  if(profiler == nullptr)
  {
    return;
  }
  profiler->flags().set(profiling::Flag::TimeMeasuredBetweenStartAndStop);
  _kernel_profilers[op_idx] = profiler;
  calculate_op_metrics(context, node_and_registration, profiler->metrics());
}

/*************************************************************************************************/
#ifdef TFLITE_MICRO_RECORDER_ENABLED
void record_layer(int op_idx, const TfLiteContext& context, const TfLiteNode &node, bool record_input)
{
  if(!model_recorder_enabled)
  {
    return;
  }

  if(record_input)
  {
      for(int i = 0; i < node.inputs->size; ++i)
      {
        TfLiteTensor *tensor = nullptr;
        const int tensor_idx = node.inputs->data[i];
        if(tensor_idx >= 0)
        {
          tensor = context.GetTensor(&context, tensor_idx);
        }
        record_tflite_tensor(tensor, op_idx, tensor_idx, true);
      }
  }
  else 
  {
      for(int i = 0; i < node.outputs->size; ++i)
      {
        TfLiteTensor *tensor = nullptr;
        const int tensor_idx = node.outputs->data[i];
        if(tensor_idx >= 0)
        {
          tensor = context.GetTensor(&context, tensor_idx);
        }
        record_tflite_tensor(tensor, op_idx, tensor_idx, false);
      }
  }

}
#endif

/*************************************************************************************************/
#ifdef TFLITE_MICRO_RECORDER_ENABLED
void record_tflite_tensor(const TfLiteTensor* tensor, int op_idx, int tensor_idx, bool is_input)
{
    TfliteMicroRecordedTensor data(tensor);
    auto& recorded_data = TfliteMicroRecordedData::instance();

    if(op_idx >= recorded_data.size())
    {
        TfliteMicroRecordedLayer layer;
        recorded_data.append(layer);
    }

    auto& layer = recorded_data[op_idx];
    if(is_input)
    {
        layer.inputs.append(data);
    }
    else 
    {
        layer.outputs.append(data);
    }
}
#endif

/*************************************************************************************************/
const char* to_str(tflite::BuiltinOperator op_type) 
{
  if (op_type == tflite::BuiltinOperator_CUSTOM) {
    return "custom";
  } else {
    return tflite::EnumNameBuiltinOperator(op_type);
  }
}

/*************************************************************************************************/
const char* op_to_str(int op_idx, tflite::BuiltinOperator op_type)
{
  static char op_name_buffer[64];
  snprintf(op_name_buffer, sizeof(op_name_buffer), "Op%d-%s", op_idx, to_str(op_type));
  return op_name_buffer;
}




} // namespace mltk