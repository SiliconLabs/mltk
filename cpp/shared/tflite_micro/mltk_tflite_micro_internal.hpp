
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_allocator.h"


#include "profiling/profiler.hpp"
#include "logging/logger.hpp"

#include "mltk_tflite_micro_helper.hpp"
#include "mltk_tflite_micro_recorder.hpp"


#define SET_CURRENT_KERNEL(op_idx, op_code) mltk::set_current_kernel(op_idx, (tflite::BuiltinOperator)op_code);
#define CLEAR_CURRENT_KERNEL() mltk::clear_current_kernel();



#ifdef TFLITE_MICRO_PROFILER_ENABLED

#define ALLOCATE_PROFILERS(subgraph_idx, operators_size) \
  FREE_PROFILERS(); mltk::allocate_profilers(subgraph_idx, operators_size);

#define REGISTER_PROFILER(subgraph_idx, op_idx, op_type, context, node_and_registration) \
   mltk::register_profiler(subgraph_idx, op_idx, (tflite::BuiltinOperator)op_type, context, node_and_registration); \
  allocator_->ResetTempAllocations();

#define FREE_PROFILERS() mltk::free_profilers();

#define START_INFERENCE_PROFILER(subgraph_idx) \
if(subgraph_idx == 0) \
{ \
  if(mltk::_inference_profiler != nullptr) mltk::_inference_profiler->start(); \
  if(mltk::mltk_tflite_micro_get_registered_accelerator() != nullptr) mltk::mltk_tflite_micro_get_registered_accelerator()->start_profiler(-1); \
}

#define STOP_INFERENCE_PROFILER(subgraph_idx) \
if(subgraph_idx == 0) \
{ \
  if(mltk::mltk_tflite_micro_get_registered_accelerator() != nullptr) mltk::mltk_tflite_micro_get_registered_accelerator()->stop_profiler(-1); \
  if(mltk::_inference_profiler != nullptr) mltk::_inference_profiler->stop(); \
}

#define START_OP_PROFILER(subgraph_idx, op_idx, op_code) \
if(subgraph_idx == 0) \
{ \
  if(mltk::_kernel_profilers != nullptr) mltk::_kernel_profilers[op_idx]->start(); \
  if(mltk::mltk_tflite_micro_get_registered_accelerator() != nullptr) mltk::mltk_tflite_micro_get_registered_accelerator()->start_op_profiler(op_idx, (mltk::_kernel_profilers != nullptr) ? mltk::_kernel_profilers[op_idx] : nullptr); \
}

#define STOP_OP_PROFILER(subgraph_idx, op_idx) \
if(subgraph_idx == 0) \
{ \
  if(mltk::mltk_tflite_micro_get_registered_accelerator() != nullptr) mltk::mltk_tflite_micro_get_registered_accelerator()->stop_op_profiler(op_idx, (mltk::_kernel_profilers != nullptr) ? mltk::_kernel_profilers[op_idx] : nullptr); \
  if(mltk::_kernel_profilers != nullptr) mltk::_kernel_profilers[op_idx]->stop(); \
}


#else // TFLITE_MICRO_PROFILER_ENABLED

#define ALLOCATE_PROFILERS(...)
#define REGISTER_PROFILER(...)
#define FREE_PROFILERS(...)
#define START_INFERENCE_PROFILER(...)
#define STOP_INFERENCE_PROFILER(...)
#define START_OP_PROFILER(subgraph_idx, op_idx, op_code)
#define STOP_OP_PROFILER(subgraph_idx, op_idx)

#endif // TFLITE_MICRO_PROFILER_ENABLED


#ifdef TFLITE_MICRO_ACCELERATOR_PROFILER_ENABLED

#undef START_INFERENCE_PROFILER
#undef STOP_INFERENCE_PROFILER

#define START_INFERENCE_PROFILER(subgraph_idx) \
  const int _accelerator_loop_count = (mltk::mltk_tflite_micro_get_registered_accelerator() != nullptr) ? mltk::mltk_tflite_micro_get_registered_accelerator()->get_profiler_loop_count() : 1; \
  for(int _accelerator_loop = 0; _accelerator_loop < _accelerator_loop_count; ++_accelerator_loop) \
  { \
  if(_accelerator_loop == 0) if(mltk::_inference_profiler != nullptr) mltk::_inference_profiler->start(); \
  if(mltk::mltk_tflite_micro_get_registered_accelerator() != nullptr) mltk::mltk_tflite_micro_get_registered_accelerator()->start_profiler(_accelerator_loop);

#define STOP_INFERENCE_PROFILER(subgraph_idx) \
  if(mltk::mltk_tflite_micro_get_registered_accelerator() != nullptr) mltk::mltk_tflite_micro_get_registered_accelerator()->stop_profiler(_accelerator_loop); \
  if(_accelerator_loop == 0) if(mltk::_inference_profiler != nullptr)  mltk::_inference_profiler->stop(); \
  }



#endif // TFLITE_MICRO_ACCELERATOR_PROFILER_ENABLED


#define INVOKE_PROCESSING_CALLBACK() \
if(mltk::_processing_callback != nullptr) mltk::_processing_callback(mltk::_processing_callback_arg)
#define INVOKE_LAYER_CALLBACK(index, context, node_and_registration, invoke_status) \
if(mltk::_layer_callback != nullptr){\
  invoke_status = mltk::_layer_callback(index, context, node_and_registration, invoke_status, mltk::_layer_callback_arg); \
}


namespace mltk
{




extern profiling::Profiler *_inference_profiler;
extern profiling::Profiler **_kernel_profilers;
extern int _current_kernel_index;
extern int _current_kernel_op_code;
extern void (*_processing_callback)(void*);
extern void* _processing_callback_arg;

typedef TfLiteStatus (*TfliteMicroLayerCallback)(
  int index,
  TfLiteContext& context,
  const tflite::NodeAndRegistration& node_and_registration,
  TfLiteStatus invoke_status,
  void* arg
);
extern TfliteMicroLayerCallback _layer_callback;
extern void *_layer_callback_arg;

void allocate_profilers(int subgraph_index, int op_count);

void register_profiler(
  int subgraph_idx,
  int op_idx,
  tflite::BuiltinOperator op_type,
  const TfLiteContext* context,
  const tflite::NodeAndRegistration& node_and_registration
);

void free_profilers();


bool calculate_op_metrics(
  const TfLiteContext* context,
  const tflite::NodeAndRegistration& node_and_registration,
  profiling::Metrics& metrics
);



static inline void set_current_kernel(int op_index, tflite::BuiltinOperator op_code)
{
  _current_kernel_index = op_index;
  _current_kernel_op_code = op_code;
  reset_unsupported_kernel_messages();
}

static inline void clear_current_kernel()
{
  _current_kernel_index = -1;
  _current_kernel_op_code = -1;
  reset_unsupported_kernel_messages();
}

static inline void set_layer_callback(TfliteMicroLayerCallback callback, void *arg)
{
  _layer_callback = callback;
  _layer_callback_arg = arg;
}

} // namespace mltk