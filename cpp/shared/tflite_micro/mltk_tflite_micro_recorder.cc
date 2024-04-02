#ifndef MLTK_DLL_IMPORT
#include <new>
#include <cassert>

#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/memory_planner/greedy_memory_planner.h"

#include "mltk_tflite_micro_recorder.hpp"
#include "mltk_tflite_micro_model_helper.hpp"


namespace mltk 
{




TfliteMicroRecorder& TfliteMicroRecorder::instance()
{
    static uint8_t instance_buffer[sizeof(TfliteMicroRecorder)];
    static TfliteMicroRecorder* instance_ptr = nullptr;

    if(instance_ptr == nullptr)
    {
        instance_ptr = new(instance_buffer)TfliteMicroRecorder();
    }

    return *instance_ptr;
}

bool TfliteMicroRecorder::is_enabled()
{
    auto& self = instance();
    return self._enabled;
}

void TfliteMicroRecorder::set_enabled(bool enabled)
{
    auto& self = instance();
    self._enabled = enabled;
}

bool TfliteMicroRecorder::is_tensor_data_recording_enabled()
{
    auto& self = instance();
    return self._tensor_data_recording_enabled;
}

void TfliteMicroRecorder::set_tensor_data_recording_enabled(bool enabled)
{
    auto& self = instance();
    if(enabled)
    {
        set_enabled(true);
    }
    self._tensor_data_recording_enabled = enabled;
}

int TfliteMicroRecorder::current_layer_index()
{
    auto& self = instance();
    assert(self._current_layer_index != -1);
    return self._current_layer_index;
}

const TfLiteNode* TfliteMicroRecorder::current_layer_node()
{
    auto& self = instance();
    assert(self._current_layer_node != nullptr);
    return self._current_layer_node;
}

void TfliteMicroRecorder::set_layer_callback(Event event, EventCallback callback)
{
    auto& self = instance();
    if(callback != nullptr && 
        !(self._layer_callbacks[(int)event] == nullptr || self._layer_callbacks[(int)event] == callback))
    {
        assert(!"Layer event callback already set");
    }
    self._layer_callbacks[(int)event] = callback;
}

bool TfliteMicroRecorder::get_recorded_data(const uint8_t** data_ptr, uint32_t* length_ptr)
{
    auto& self = instance();
    if(self._state != State::Finished)
    {
       return false;
    }

    return msgpack_buffered_writer_get_buffer(self._msgpack, (uint8_t**)data_ptr, length_ptr) == 0;
}

void TfliteMicroRecorder::reset()
{
    auto& self = instance();
    msgpack_buffered_writer_deinit(self._msgpack, true);
    self._msgpack = nullptr;
    self._state = State::Idle;
    self._current_layer_index = -1;
    self._current_layer_node = nullptr;
    self._section_active = false;
    self._layer_active = false;
}

bool TfliteMicroRecorder::start()
{
    auto& self = instance();

    if(!self._enabled)
    {
        return false;
    }
    else if(self._state != State::Idle)
    {
       assert(!"Bad recording state, not idle");
       return false;
    }
    else if(msgpack_buffered_writer_init(&self._msgpack, 32*1024) != 0)
    {
        return false;
    }

    msgpack_write_dict_marker(self._msgpack, -1);
    self._state = State::Started;

    return true;
}

bool TfliteMicroRecorder::begin_section()
{
    auto& self = instance();

    if(self._state == State::Idle || self._state == State::Finished)
    {
       return false;
    }

    assert(!self._section_active);

    if(self._state == State::Started)
    {
        self._state = State::RecordingInitSection;
        self._section_active = true;
        return msgpack_write_dict_array(self._msgpack, "init", -1) == 0;
    }
    else if(self._state == State::RecordingInitSection)
    {
        self._state = State::RecordingPrepareSection;
        self._section_active = true;
        return msgpack_write_dict_array(self._msgpack, "prepare", -1) == 0;
    }
    else if(self._state == State::RecordingPrepareSection)
    {
        self._state = State::RecordingExecutionSection;
        self._section_active = true;
        return msgpack_write_dict_array(self._msgpack, "execute", -1) == 0;
    }
    else 
    {
        assert(!"Unsupported state");
        return false;
    }
}

bool TfliteMicroRecorder::end_section()
{
    auto& self = instance();

    if(self._state == State::Idle || self._state == State::Finished)
    {
       return false;
    }

    assert(self._section_active);
    assert(self._state >= State::RecordingInitSection && self._state <= State::RecordingExecutionSection);
    assert(self._current_layer_index == -1);

    self._section_active = false;

    // Finish the section layers array
    msgpack_finalize_dynamic(self._msgpack);

    if(self._state == State::RecordingExecutionSection)
    {
        self._state = State::Finished;
        // Finish the model dictionary
        msgpack_finalize_dynamic(self._msgpack);
    }

    return true;
}

bool TfliteMicroRecorder::begin_layer(TfLiteContext* context, int index, const TfLiteNode *node)
{
    auto& self = instance();

    if(self._state == State::Idle || self._state == State::Finished)
    {
       return false;
    }

    assert(!self._layer_active);
    assert(self._state >= State::RecordingInitSection && self._state <= State::RecordingExecutionSection);
    assert(self._current_layer_index == -1);

    self._current_layer_index = index;
    self._current_layer_node = node;

    if(self._state == State::RecordingPrepareSection)
    {
        if(self._layer_callbacks[(int)Event::PrepareBegin] != nullptr)
        {
            self._layer_callbacks[(int)Event::PrepareBegin]();
        }
    }
    else if(self._state == State::RecordingExecutionSection)
    {
        _record_layer_input_tensors(context);
        if(self._layer_callbacks[(int)Event::ExecutionBegin] != nullptr)
        {
            self._layer_callbacks[(int)Event::ExecutionBegin]();
        }
    }

    return true;
}

bool TfliteMicroRecorder::end_layer(TfLiteContext* context)
{
    auto& self = instance();

    if(self._state == State::Idle || self._state == State::Finished)
    {
       return false;
    }

    if(self._state == State::RecordingPrepareSection)
    {
        if(self._layer_callbacks[(int)Event::PreparedEnd] != nullptr)
        {
            self._layer_callbacks[(int)Event::PreparedEnd]();
        }
    }
    else if(self._state == State::RecordingExecutionSection)
    {
        _record_layer_output_tensors(context);
        if(self._layer_callbacks[(int)Event::ExecutionEnd] != nullptr)
        {
            self._layer_callbacks[(int)Event::ExecutionEnd]();
        }
    }

    if(self._layer_active)
    {
        self._layer_active = false;
        // Finish the layer dictionary
        msgpack_finalize_dynamic(self._msgpack);
    }

    self._current_layer_index = -1;
    self._current_layer_node = nullptr;

    return true;
}

msgpack_context_t* TfliteMicroRecorder::get_context()
{
    auto& self = instance();
    if(self._state == State::Idle)
    {
        return nullptr;
    }

    if(self._section_active && self._state >= State::RecordingInitSection && self._state <= State::RecordingExecutionSection)
    {
        if(!self._layer_active)
        {
            assert(self._current_layer_index != -1);
            self._layer_active = true;
            // Start the layer dictionary
            msgpack_write_dict_marker(self._msgpack, -1);
            msgpack_write_dict_int(self._msgpack, "index", self._current_layer_index);
        }
    }

    return self._msgpack;
}

void TfliteMicroRecorder::_record_layer_input_tensors(TfLiteContext* context)
{
    auto& self = instance();

    if(!self._tensor_data_recording_enabled)
    {
        return;
    }

    auto& node = *self._current_layer_node;

    // Active the current layer
    get_context();

    msgpack_write_dict_array(self._msgpack, "inputs", node.inputs->size);

    for(int i = 0; i < node.inputs->size; ++i)
    {
        const int tensor_idx = node.inputs->data[i];

        if(tensor_idx >= 0)
        {
            size_t tensor_size_bytes;
            auto eval_tensor = context->GetEvalTensor(context, tensor_idx);
            tflite::TfLiteEvalTensorByteLength(eval_tensor, &tensor_size_bytes);
            msgpack_write_bin(self._msgpack, eval_tensor->data.raw, tensor_size_bytes);
        }
        else
        {
            msgpack_write_nil(self._msgpack);
        }
    }
}

void TfliteMicroRecorder::_record_layer_output_tensors(TfLiteContext* context)
{
    auto& self = instance();

    if(!self._tensor_data_recording_enabled)
    {
        return;
    }

    auto& node = *self._current_layer_node;

    msgpack_write_dict_array(self._msgpack, "outputs", node.outputs->size);

    for(int i = 0; i < node.outputs->size; ++i)
    {
        const int tensor_idx = node.outputs->data[i];
        if(tensor_idx >= 0)
        {
            size_t tensor_size_bytes;
            auto eval_tensor = context->GetEvalTensor(context, tensor_idx);
            tflite::TfLiteEvalTensorByteLength(eval_tensor, &tensor_size_bytes);
            msgpack_write_bin(self._msgpack, eval_tensor->data.raw, tensor_size_bytes);
        }
        else
        {
            msgpack_write_nil(self._msgpack);
        }
    }
}


static void get_tensor_id(
    const tflite::AllocationInfo* allocation_info,
    size_t* allocation_offsets, 
    const tflite::AllocationInfo* info,
    int* subgraph_id,
    int* tensor_id
)
{
    *subgraph_id = -1;
    *tensor_id = -1;

    const auto model = TfliteMicroModelHelper::model(TfliteMicroModelHelper::active_tflite_context());
    for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs()->size();
        subgraph_idx++) 
    {
        const auto subgraph_allocation_info = &allocation_info[allocation_offsets[subgraph_idx]];
        const auto subgraph = model->subgraphs()->Get(subgraph_idx);
        for (size_t i = 0; i < subgraph->tensors()->size(); ++i) 
        {
            const auto current = &subgraph_allocation_info[i];
            if((uintptr_t)current == (uintptr_t)info)
            {
                *subgraph_id = subgraph_idx;
                *tensor_id = i;
                return;
            }
        }
    }
}

bool TfliteMicroRecorder::record_memory_plan(
    tflite::MicroMemoryPlanner* memory_planner,
    const tflite::AllocationInfo* allocation_info,
    int allocation_info_length,
    size_t* allocation_offsets
)
{
  auto& self = instance();
  msgpack_context_t *msgpack = get_context();
  if(msgpack == nullptr)
  {
    return false;
  }

  const int buffer_count = memory_planner->GetBufferCount();

  msgpack_write_dict_array(msgpack, "memory_plan", buffer_count);
  int buffer_index = 0;
  for (int i = 0; i < allocation_info_length; ++i) 
  {
    const auto& info = allocation_info[i];
    if(!info.needs_allocating)
    {
        continue;
    }

    int offset;
    assert(buffer_index < buffer_count);
    memory_planner->GetOffsetForBuffer(buffer_index, &offset);
    buffer_index++;

    msgpack_write_dict_marker(msgpack, 6);
    msgpack_write_dict_int(msgpack, "size", info.bytes);
    msgpack_write_dict_int(msgpack, "offset", offset);
    msgpack_write_dict_int(msgpack, "start", info.first_created);
    msgpack_write_dict_int(msgpack, "end", info.last_used);

    int subgraph_id, tensor_id;
    get_tensor_id(allocation_info, allocation_offsets, &info, &subgraph_id, &tensor_id);
    msgpack_write_dict_int(msgpack, "subgraph_id", subgraph_id);
    msgpack_write_dict_int(msgpack, "tensor_id", tensor_id);
  }

  return true;
}

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

bool TfliteMicroRecorder::record_layer_conv_params(
    const tflite::ConvParams& params, 
    const int32_t* per_channel_output_multiplier, 
    const int32_t *per_channel_output_shift,
    int n_channels
)
{
  msgpack_context_t *msgpack = get_context();
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

  return true;
}

bool TfliteMicroRecorder::record_layer_depthwise_conv_params(
    const tflite::DepthwiseParams& params, 
    const int32_t* per_channel_output_multiplier, 
    const int32_t *per_channel_output_shift,
    int n_channels
)
{
  msgpack_context_t *msgpack = get_context();
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

  return true;
}

bool TfliteMicroRecorder::record_layer_fully_connected_params(
    const tflite::FullyConnectedParams& params
)
{
  msgpack_context_t *msgpack = get_context();
  msgpack_write_dict_dict(msgpack, "params", -1);
  msgpack_write_dict_int(msgpack, "input_offset", params.input_offset);
  msgpack_write_dict_int(msgpack, "weights_offset", params.weights_offset);
  msgpack_write_dict_int(msgpack, "output_offset", params.output_offset);
  msgpack_write_dict_int(msgpack, "output_multiplier", params.output_multiplier);
  msgpack_write_dict_int(msgpack, "output_shift", params.output_shift);
  msgpack_write_dict_int(msgpack, "quantized_activation_min", params.quantized_activation_min);
  msgpack_write_dict_int(msgpack, "quantized_activation_max", params.quantized_activation_max);
  msgpack_finalize_dynamic(msgpack);

  return true;
}

bool TfliteMicroRecorder::record_layer_pool_params(
    const tflite::PoolParams& params
)
{
  msgpack_context_t *msgpack = get_context();
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

  return true;
}

} // namespace mltk 


#endif // MLTK_DLL_IMPORT