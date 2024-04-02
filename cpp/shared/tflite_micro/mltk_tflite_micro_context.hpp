#pragma once 

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_interpreter_context.h"

namespace mltk
{

typedef TfLiteStatus (*TfliteMicroLayerCallback)(
  int index,
  TfLiteContext& context,
  const tflite::NodeAndRegistration& node_and_registration,
  TfLiteStatus invoke_status,
  void* arg
);

typedef void (*TfliteMicroProcessingCallback)(void *arg);

class TfliteMicroAccelerator;


class TfliteMicroContext
{
public:

  static TfliteMicroContext* create(TfLiteContext *context)
  {
    auto buffer = context->AllocatePersistentBuffer(context, sizeof(TfliteMicroContext));
    if(buffer == nullptr)
    {
      return nullptr;
    }
    return new(buffer)TfliteMicroContext();
  }

  virtual bool init(
    const void* flatbuffer,
    TfLiteContext *context,
    TfliteMicroAccelerator* accelerator,
    tflite::MicroAllocator* allocator
  )
  {
    this->flatbuffer = flatbuffer;
    this->context = context;
    this->accelerator = accelerator;
    this->allocator = allocator;

    auto micro_context = tflite::GetMicroContext(context);
    assert(micro_context != nullptr);
    auto interpreter_micro_context = reinterpret_cast<tflite::MicroInterpreterContext*>(micro_context);
    auto state = interpreter_micro_context->GetInterpreterState();
    interpreter_micro_context->SetInterpreterState(tflite::MicroInterpreterContext::InterpreterState::kPrepare);
    micro_context->set_external_context(this);
    interpreter_micro_context->SetInterpreterState(state);
    return true;
  }

  virtual bool load(TfLiteContext *context) { return true; }

  TfLiteContext *context = nullptr;
  const void* flatbuffer = nullptr;
  TfliteMicroAccelerator* accelerator = nullptr;
  tflite::MicroAllocator* allocator = nullptr;
  tflite::BuiltinOperator current_layer_opcode = (tflite::BuiltinOperator)(-1);
  int current_layer_index = -1;
  TfliteMicroLayerCallback layer_callback = nullptr;
  void *layer_callback_arg = nullptr;
  TfliteMicroProcessingCallback processing_callback = nullptr;
  void *processing_callback_arg = nullptr;

protected:
  TfliteMicroContext() = default;

};




} // namespace mltk