#pragma once 

#include <cassert>





#include "cpputils/helpers.hpp"
#include "mltk_tflite_micro_context.hpp"
#include "mltk_tflite_micro_context.hpp"


namespace mltk
{




class DLL_EXPORT TfliteMicroModelHelper
{
public:
    static const char* const TFLITE_MICRO_VERSION;

    static const tflite::Model* model(TfLiteContext* context=nullptr);
    static TfliteMicroContext* mltk_tflite_micro_context(TfLiteContext* context=nullptr);
    static tflite::MicroContext* tflite_micro_context(TfLiteContext* context=nullptr);
    static tflite::MicroAllocator* tflite_micro_allocator(TfLiteContext* context=nullptr);
    static TfliteMicroAccelerator* tflite_micro_accelerator(TfLiteContext* context=nullptr);
    static tflite::BuiltinOperator current_layer_opcode(TfLiteContext* context=nullptr);
    static int current_layer_index(TfLiteContext* context=nullptr);
    static const char* current_layer_name(TfLiteContext* context=nullptr);
    static void set_current_layer(TfLiteContext* context, int index, tflite::BuiltinOperator opcode);
    static void clear_current_layer(TfLiteContext* context);
    
    static void set_layer_callback(
      TfLiteContext* context, 
      TfliteMicroLayerCallback callback, 
      void *arg = nullptr
    );
    static TfLiteStatus invoke_layer_callback(
      TfLiteContext* context,
      const tflite::NodeAndRegistration& node_and_registration,
      TfLiteStatus invoke_status
    );
    static void set_processing_callback(
      TfLiteContext* context, 
      TfliteMicroProcessingCallback callback, 
      void *arg = nullptr)
    ;
    static void invoke_processing_callback(TfLiteContext* context);
    
    template<typename T = uint8_t>
    static  T* allocate_persistent_buffer(
      TfLiteContext* context, 
      unsigned count
    )
    {
      if(context == nullptr) return nullptr;
      return reinterpret_cast<T*>(context->AllocatePersistentBuffer(context, sizeof(T) * count));
    }

    template<typename T = uint8_t>
    static  T* allocate_planned_persistent_buffer(
      TfLiteContext* context, 
      unsigned count
    )
    {
      T* retval = nullptr;
      auto allocator = tflite_micro_allocator(context);

      if(allocator != nullptr)
      {
        retval = reinterpret_cast<T*>(allocator->allocate_planned_persistent_buffer(count * sizeof(T)));
      }

      if(retval == nullptr)
      {
        retval = allocate_persistent_buffer<T>(context, count);
      }

      return retval;
    }


    template<typename T = uint8_t>
    static TfLiteStatus allocate_scratch_buffer(
      TfLiteContext* context, 
      unsigned count, 
      int *buffer_index
    )
    {
        return context->RequestScratchBufferInArena(context, sizeof(T)*count, buffer_index);
    }

    template<typename T = uint8_t>
    static T* get_scratch_buffer(TfLiteContext* context, int scratch_buffer_index)
    {
        return reinterpret_cast<T*>(context->GetScratchBuffer(context, scratch_buffer_index));
    }


    static TfLiteContext* active_tflite_context();
    static void set_active_tflite_context(TfLiteContext* context);

    static const char* opcode_to_str(tflite::BuiltinOperator opcode);
    static const char* create_layer_name(int layer_idx, tflite::BuiltinOperator opcode);
    static bool verify_model_flatbuffer(const void* flatbuffer, int flatbuffer_length);
    static const void* get_metadata_from_tflite_flatbuffer(
      const void* tflite_flatbuffer, 
      const char* tag, 
      uint32_t* length = nullptr
    );
    static const void* get_metadata_from_tflite_flatbuffer(
      TfLiteContext* context, 
      const char* tag, 
      uint32_t* length = nullptr
    );
    static const void* get_metadata_from_tflite_flatbuffer(
      const tflite::Model *model, 
      const char* tag, 
      uint32_t* length = nullptr
    );

    static bool get_tflite_flatbuffer_from_end_of_flash(
      const uint8_t** tflite_flatbuffer, 
      uint32_t* length=nullptr, 
      const uint32_t* flash_end_addr=nullptr
    );
};



#define MLTK_ALLOCATE_PERSISTENT_BUFFER(type, count) \
  ::mltk::TfliteMicroModelHelper::allocate_persistent_buffer<type>(context, count)
#define MLTK_ALLOCATE_PLANNED_PERSISTENT_BUFFER(type, count) \
  ::mltk::TfliteMicroModelHelper::allocate_planned_persistent_buffer<type>(context, count)
#define MLTK_ALLOCATE_SCRATCH_BUFFER(size_bytes, scratch_buffer_index) \
  ::mltk::TfliteMicroModelHelper::allocate_scratch_buffer(context, size_bytes, scratch_buffer_index)
#define MLTK_GET_SCRATCH_BUFFER(type, scratch_buffer_index) \
  ::mltk::TfliteMicroModelHelper::get_scratch_buffer<type>(context, scratch_buffer_index)

#define MLTK_SET_CURRENT_KERNEL(op_idx, op_code) \
  ::mltk::TfliteMicroModelHelper::set_current_layer(context, op_idx, (tflite::BuiltinOperator)op_code)
#define MLTK_CLEAR_CURRENT_KERNEL() \
  ::mltk::TfliteMicroModelHelper::clear_current_layer(context)

#define MLTK_INVOKE_LAYER_CALLBACK(node_and_registration, invoke_status) \
  invoke_status = ::mltk::TfliteMicroModelHelper::invoke_layer_callback(context, node_and_registration, invoke_status)

#define MLTK_INVOKE_PROCESSING_CALLBACK() \
 ::mltk::TfliteMicroModelHelper::invoke_processing_callback(context)



} // namespace mltk