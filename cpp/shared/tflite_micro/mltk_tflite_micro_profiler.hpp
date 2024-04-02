#pragma once 


#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "cpputils/helpers.hpp"
#include "profiling/profiling.hpp"
#include "mltk_tflite_micro_accelerator.hpp"



namespace mltk 
{

class DLL_EXPORT TfliteMicroProfiler 
{
public:
    static TfliteMicroProfiler& instance();

    static void set_enabled(bool enabled);
    static bool is_enabled();

    static void init(int count);
    static void deinit();
    static void register_profiler(
        TfLiteContext* context,
        int layer_index,
        tflite::BuiltinOperator opcode,
        const tflite::NodeAndRegistration& node_and_registration
    );

    static void start(int loop_index = -1);
    static void stop(int loop_index = -1);
    static void start_layer(int index);
    static void stop_layer(int index);

    static bool calculate_op_metrics(
        TfLiteContext *context,
        const tflite::NodeAndRegistration& node_and_registration,
        profiling::Metrics& metrics
    );

private:
    bool _enabled = false;
    profiling::Profiler *_inference_profiler = nullptr;
    profiling::Profiler **_layer_profilers = nullptr;

    TfliteMicroProfiler() = default;
};




#ifdef TFLITE_MICRO_PROFILER_ENABLED

#define MLTK_ALLOCATE_PROFILERS(subgraph_idx, operators_size) \
  if(subgraph_idx == 0) { \
    MLTK_FREE_PROFILERS(); \
    ::mltk::TfliteMicroProfiler::init(operators_size); \
  }

#define MLTK_FREE_PROFILERS() \
    ::mltk::TfliteMicroProfiler::deinit()

#define MLTK_REGISTER_PROFILER(subgraph_idx, op_idx, op_type, node_and_registration) \
   if(subgraph_idx == 0) { \
    ::mltk::TfliteMicroProfiler::register_profiler(\
        context, \
        op_idx, \
        (tflite::BuiltinOperator)op_type, \
        node_and_registration \
    ); \
    allocator_->ResetTempAllocations(); \
   }
  
#define MLTK_START_INFERENCE_PROFILER(subgraph_idx) \
    if(subgraph_idx == 0) { \
        ::mltk::TfliteMicroProfiler::start(); \
    }

#define MLTK_STOP_INFERENCE_PROFILER(subgraph_idx) \
    if(subgraph_idx == 0) { \
        ::mltk::TfliteMicroProfiler::stop(); \
    }

#define MLTK_START_OP_PROFILER(subgraph_idx, op_idx) \
    if(subgraph_idx == 0) { \
        ::mltk::TfliteMicroProfiler::start_layer(op_idx); \
    }

#define MLTK_STOP_OP_PROFILER(subgraph_idx, op_idx) \
    if(subgraph_idx == 0) { \
        ::mltk::TfliteMicroProfiler::stop_layer(op_idx); \
    }


#else 

#define MLTK_ALLOCATE_PROFILERS(...)
#define MLTK_FREE_PROFILERS(...)
#define MLTK_REGISTER_PROFILER(...)

#define MLTK_START_INFERENCE_PROFILER(...)
#define MLTK_STOP_INFERENCE_PROFILER(...)
#define MLTK_START_OP_PROFILER(subgraph_idx, op_idx)
#define MLTK_STOP_OP_PROFILER(subgraph_idx, op_idx)


#endif // TFLITE_MICRO_PROFILER_ENABLED


#ifdef TFLITE_MICRO_ACCELERATOR_PROFILER_ENABLED

#undef MLTK_START_INFERENCE_PROFILER
#undef MLTK_STOP_INFERENCE_PROFILER

#define MLTK_START_INFERENCE_PROFILER(subgraph_idx) \
    const auto _accelerator = mltk::mltk_tflite_micro_get_registered_accelerator(); \
    const int _accelerator_loop_count = (_accelerator != nullptr) ? _accelerator->get_profiler_loop_count() : 1; \
    for(int _accelerator_loop = 0; _accelerator_loop < _accelerator_loop_count; ++_accelerator_loop) { \
        ::mltk::TfliteMicroProfiler::start(_accelerator_loop); \


#define MLTK_STOP_INFERENCE_PROFILER(subgraph_idx) \
    ::mltk::TfliteMicroProfiler::stop(_accelerator_loop); \
    }

#endif // TFLITE_MICRO_ACCELERATOR_PROFILER_ENABLED


} // namespace mltk 