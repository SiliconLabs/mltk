#pragma once 
#include <cstdint>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/micro/memory_planner/micro_memory_planner.h"
#include "tensorflow/lite/micro/micro_allocation_info.h"
#include "cpputils/helpers.hpp"
#include "msgpack.hpp"


namespace mltk
{

class DLL_EXPORT TfliteMicroRecorder 
{
public:
  enum class Event : uint8_t
  {
    PrepareBegin,
    PreparedEnd,
    ExecutionBegin,
    ExecutionEnd,
    Count
  };

  typedef void (*EventCallback)();


  static TfliteMicroRecorder& instance();
  static bool is_enabled();
  static void set_enabled(bool enabled);
  static bool is_tensor_data_recording_enabled();
  static void set_tensor_data_recording_enabled(bool enabled);
  static int current_layer_index();
  static const TfLiteNode* current_layer_node();
  static void set_layer_callback(Event event, EventCallback callback);

  static msgpack_context_t* get_context();
  static bool get_recorded_data(const uint8_t** data_ptr, uint32_t* length_ptr);

  static void reset();
  static bool start();
  static bool begin_section();
  static bool end_section();
  static bool begin_layer(TfLiteContext* context, int index, const TfLiteNode *node);
  static bool end_layer(TfLiteContext* context);
  

  static bool record_memory_plan(
    tflite::MicroMemoryPlanner* memory_planner,
    const tflite::AllocationInfo* allocation_info,
    int allocation_info_length,
    size_t* allocation_offsets
  );

  static bool record_layer_conv_params(
    const tflite::ConvParams& params, 
    const int32_t* per_channel_output_multiplier, 
    const int32_t *per_channel_output_shift,
    int n_channels
  );
  static bool record_layer_depthwise_conv_params(
    const tflite::DepthwiseParams& params, 
    const int32_t* per_channel_output_multiplier, 
    const int32_t *per_channel_output_shift,
    int n_channels
  );
  static bool record_layer_fully_connected_params(
    const tflite::FullyConnectedParams& params
  );
  static bool record_layer_pool_params(
    const tflite::PoolParams& params
  );


  template<typename T>
  static bool record_layer_param(
      const char *key,
      T value
  )
  {
      auto& self = instance();
      if(!self._layer_active)
      {
        return false;
      }

      msgpack_context_t *msgpack = get_context();
      if(msgpack == nullptr)
      {
          return false;
      }

      msgpack_write_dict(msgpack, key, value);

      return true;
  }

  template<typename T>
  static bool record_param(
      const char *key,
      T value
  )
  {
      auto& self = instance();
      if(self._layer_active)
      {
        return false;
      }

      msgpack_context_t *msgpack = get_context();
      if(msgpack == nullptr)
      {
          return false;
      }

      msgpack_write_dict(msgpack, key, value);

      return true;
  }

private:
  msgpack_context_t* _msgpack = nullptr;
  bool _enabled = false; 
  bool _tensor_data_recording_enabled = false;

  int _current_layer_index = -1;
  const TfLiteNode* _current_layer_node = nullptr;

  EventCallback _layer_callbacks[(int)Event::Count] = { 0 };
  bool _section_active = false;
  bool _layer_active = false;

  enum class State : uint8_t 
  {
    Idle,
    Started,
    RecordingInitSection,
    RecordingPrepareSection,
    RecordingExecutionSection,
    Finished
  };
  State _state = State::Idle;

  TfliteMicroRecorder() = default;


  static void _record_layer_input_tensors(TfLiteContext* context);
  static void _record_layer_output_tensors(TfLiteContext* context);
};


} // namespace mltk


#ifdef TFLITE_MICRO_RECORDER_ENABLED

#define MLTK_RECORD_START() ::mltk::TfliteMicroRecorder::reset(); ::mltk::TfliteMicroRecorder::start()
#define MLTK_RECORD_RESET() ::mltk::TfliteMicroRecorder::reset()
#define MLTK_RECORD_BEGIN_SECTION() ::mltk::TfliteMicroRecorder::begin_section()
#define MLTK_RECORD_END_SECTION() ::mltk::TfliteMicroRecorder::end_section()
#define MLTK_RECORD_BEGIN_LAYER(index, node) ::mltk::TfliteMicroRecorder::begin_layer(context, index, node)
#define MLTK_RECORD_END_LAYER() ::mltk::TfliteMicroRecorder::end_layer(context)

#define MLTK_RECORD_MEMORY_PLAN(planner, allocation_info, allocation_info_length, allocation_offsets) \
  ::mltk::TfliteMicroRecorder::record_memory_plan(planner, allocation_info, allocation_info_length, allocation_offsets)
#define MLTK_RECORD_CONV_PARAMS(op_params, per_channel_output_multiplier, per_channel_output_shift, n_channels) \
  ::mltk::TfliteMicroRecorder::record_layer_conv_params(op_params, per_channel_output_multiplier, per_channel_output_shift, n_channels)
#define MLTK_RECORD_DEPTHWISE_CONV_PARAMS(op_params, per_channel_output_multiplier, per_channel_output_shift, n_channels) \
  ::mltk::TfliteMicroRecorder::record_layer_depthwise_conv_params(op_params, per_channel_output_multiplier, per_channel_output_shift, n_channels)
#define MLTK_RECORD_FULLY_CONNECTED_PARAMS(op_params) ::mltk::TfliteMicroRecorder::record_layer_fully_connected_params(op_params)
#define MLTK_RECORD_POOL_PARAMS(op_params) ::mltk::TfliteMicroRecorder::record_layer_pool_params(op_params)
#define MLTK_RECORD_LAYER_PARAM(key, value) ::mltk::TfliteMicroRecorder::record_layer_param(key, value)
#define MLTK_RECORD_PARAM(key, value) ::mltk::TfliteMicroRecorder::record_param(key, value)

#else 

#define MLTK_RECORD_START()
#define MLTK_RECORD_RESET() 
#define MLTK_RECORD_BEGIN_SECTION()
#define MLTK_RECORD_END_SECTION()
#define MLTK_RECORD_BEGIN_LAYER(index, node)
#define MLTK_RECORD_END_LAYER()
#define MLTK_RECORD_MEMORY_PLAN(lanner, allocation_info, allocation_info_length, allocation_offsets)
#define MLTK_RECORD_CONV_PARAMS(op_params, per_channel_output_multiplier, per_channel_output_shift, n_channels)
#define MLTK_RECORD_DEPTHWISE_CONV_PARAMS(op_params, per_channel_output_multiplier, per_channel_output_shift, n_channels)
#define MLTK_RECORD_FULLY_CONNECTED_PARAMS(op_params)
#define MLTK_RECORD_POOL_PARAMS(op_params)
#define MLTK_RECORD_LAYER_PARAM(key, value)
#define MLTK_RECORD_PARAM(key, value)

#endif // TFLITE_MICRO_RECORDER_ENABLED


