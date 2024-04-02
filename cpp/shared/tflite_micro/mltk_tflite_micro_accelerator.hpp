#pragma once 

#include <functional>
#include "tensorflow/lite/micro/micro_allocator.h"
#include "profiling/profiler.hpp"
#include "cpputils/helpers.hpp"
#include "mltk_tflite_micro_context.hpp"


namespace mltk
{


class TfliteMicroAccelerator
{
public:
  virtual const char* name() const = 0;
  virtual bool init(){ return true; }
  virtual void deinit(TfLiteContext *context){}
  virtual int  get_profiler_loop_count() { return 0; }
  virtual void start_profiler(int loop_count){}
  virtual void stop_profiler(int loop_count){};
  virtual void start_op_profiler(int op_idx, profiling::Profiler* profiler){};
  virtual void stop_op_profiler(int op_idx, profiling::Profiler* profiler){};
  virtual bool set_simulator_memory(const char* region, void* base_address, uint32_t length){ return false; };
  virtual bool invoke_simulator(const std::function<bool()>&func){ return false; };
  
  virtual bool create_allocator(
    const void* flatbuffer,
    uint8_t* buffers[], 
    const int32_t buffer_sizes[], 
    int32_t buffer_count,
    tflite::MicroAllocator** allocator_ptr
  ) 
  { 
    *allocator_ptr = nullptr; 
    return true; 
  }

  virtual TfliteMicroContext* create_context(
    TfLiteContext *context
  )
  { 
    return TfliteMicroContext::create(context);
  }

};


extern "C" TfliteMicroAccelerator* mltk_tflite_micro_register_accelerator();
extern "C" DLL_EXPORT TfliteMicroAccelerator* mltk_tflite_micro_set_accelerator(TfliteMicroAccelerator* accelerator);
extern "C" DLL_EXPORT TfliteMicroAccelerator* mltk_tflite_micro_get_registered_accelerator();


} // namespace mltk