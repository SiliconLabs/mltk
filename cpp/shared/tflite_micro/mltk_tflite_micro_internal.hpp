
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_allocator.h"


#include "profiling/profiler.hpp"
#include "logging/logger.hpp"

#include "mltk_tflite_micro_helper.hpp"



#define SET_CURRENT_KERNEL(op_idx, op_code) \
mltk::_current_kernel_index = op_idx; \
mltk::_current_kernel_op_code = op_code; \
mltk::_issued_unsupported_msg = false;
#define CLEAR_CURRENT_KERNEL() \
mltk::_current_kernel_index = -1; \
mltk::_current_kernel_op_code = -1;



#ifdef TFLITE_MICRO_PROFILER_ENABLED

#define ALLOCATE_PROFILERS(subgraph_idx, operators_size) \
  mltk::allocate_profilers(subgraph_idx, operators_size);

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
SET_CURRENT_KERNEL(op_idx, op_code) \
if(subgraph_idx == 0) \
{ \
  if(mltk::_kernel_profilers != nullptr) mltk::_kernel_profilers[op_idx]->start(); \
  if(mltk::mltk_tflite_micro_get_registered_accelerator() != nullptr) mltk::mltk_tflite_micro_get_registered_accelerator()->start_op_profiler(op_idx, (mltk::_kernel_profilers != nullptr) ? mltk::_kernel_profilers[op_idx] : nullptr); \
}

#define STOP_OP_PROFILER(subgraph_idx, op_idx) \
CLEAR_CURRENT_KERNEL() \
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
#define START_OP_PROFILER(subgraph_idx, op_idx, op_code) SET_CURRENT_KERNEL(op_idx, op_code)
#define STOP_OP_PROFILER(subgraph_idx, op_idx) CLEAR_CURRENT_KERNEL()

#endif // TFLITE_MICRO_PROFILER_ENABLED



#ifdef TFLITE_MICRO_RECORDER_ENABLED

#define RECORD_INPUTS(op_idx, context, node) mltk::record_layer(op_idx, *context, *node, true);
#define RECORD_OUTPUTS(op_idx, context, node)  mltk::record_layer(op_idx, *context, *node, false);

#else 

#define RECORD_INPUTS(...)
#define RECORD_OUTPUTS(...)

#endif // TFLITE_MICRO_RECORDER_ENABLED


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



namespace mltk
{




extern profiling::Profiler *_inference_profiler;
extern profiling::Profiler **_kernel_profilers;
extern int _current_kernel_index;
extern int _current_kernel_op_code;
extern bool _issued_unsupported_msg;
extern void (*_processing_callback)(void*);
extern void* _processing_callback_arg;


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


void record_layer(int op_idx, const TfLiteContext& context, const TfLiteNode &node, bool record_input);
void record_tflite_tensor(const TfLiteTensor* tensor, int op_idx, int tensor_idx, bool is_input);

const char* to_str(tflite::BuiltinOperator op_type);
const char* op_to_str(int op_idx, tflite::BuiltinOperator op_type);


} // namespace mltk