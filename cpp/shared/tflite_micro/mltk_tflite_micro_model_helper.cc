#ifndef MLTK_DLL_IMPORT
#include <new>
#include <cstdio>

#include "tensorflow/lite/schema/schema_generated.h"

#include "em_device.h"
#include "mltk_tflite_micro_model_helper.hpp"
#include "mltk_tflite_micro_logger.hpp"


namespace mltk 
{

#ifndef TFLITE_MICRO_VERSION_STR
#define TFLITE_MICRO_VERSION_STR nullptr
#endif
const char* const TfliteMicroModelHelper::TFLITE_MICRO_VERSION = TFLITE_MICRO_VERSION_STR;

const tflite::Model* TfliteMicroModelHelper::model(TfLiteContext* context)
{
    auto& mltk_micro_context = *mltk_tflite_micro_context(context);
    if(mltk_micro_context.flatbuffer == nullptr)
    {
        return nullptr;
    }
    return tflite::GetModel(mltk_micro_context.flatbuffer);
}

TfliteMicroContext* TfliteMicroModelHelper::mltk_tflite_micro_context(TfLiteContext* context)
{
    auto micro_context = tflite_micro_context(context);
    auto mltk_context = reinterpret_cast<TfliteMicroContext*>(micro_context->external_context());
    assert(mltk_context != nullptr);
    return mltk_context;
}

tflite::MicroContext* TfliteMicroModelHelper::tflite_micro_context(TfLiteContext* context)
{
    context = (context == nullptr) ? active_tflite_context() : context;
    assert(context != nullptr);
    auto tflm_context = tflite::GetMicroContext(context);
    assert(tflm_context != nullptr);
    return tflm_context;
}

tflite::MicroAllocator* TfliteMicroModelHelper::tflite_micro_allocator(TfLiteContext* context)
{
    auto& mltk_micro_context = *mltk_tflite_micro_context(context);
    return mltk_micro_context.allocator;
}

TfliteMicroAccelerator* TfliteMicroModelHelper::tflite_micro_accelerator(TfLiteContext* context)
{
    auto& mltk_micro_context = *mltk_tflite_micro_context(context);
    return mltk_micro_context.accelerator;
}

tflite::BuiltinOperator TfliteMicroModelHelper::current_layer_opcode(TfLiteContext* context)
{
    auto& mltk_micro_context = *mltk_tflite_micro_context(context);
    return mltk_micro_context.current_layer_opcode;
}

int TfliteMicroModelHelper::current_layer_index(TfLiteContext* context)
{
    auto& mltk_micro_context = *mltk_tflite_micro_context(context);
    return mltk_micro_context.current_layer_index;
}
const char* TfliteMicroModelHelper::current_layer_name(TfLiteContext* context)
{
    auto& mltk_micro_context = *mltk_tflite_micro_context(context);
    return create_layer_name(
        mltk_micro_context.current_layer_index,
        mltk_micro_context.current_layer_opcode
    );
}

void TfliteMicroModelHelper::set_current_layer(
    TfLiteContext* context,
    int index, 
    tflite::BuiltinOperator opcode
)
{
    auto& mltk_micro_context = *mltk_tflite_micro_context(context);
    mltk_micro_context.current_layer_index = index;
    mltk_micro_context.current_layer_opcode = opcode;
}


void TfliteMicroModelHelper::clear_current_layer(TfLiteContext* context)
{
    auto& mltk_micro_context = *mltk_tflite_micro_context(context);
    mltk_micro_context.current_layer_opcode = (tflite::BuiltinOperator)(-1);
    mltk_micro_context.current_layer_index = -1;
}


void TfliteMicroModelHelper::set_layer_callback(
    TfLiteContext* context, 
    TfliteMicroLayerCallback callback, 
    void *arg
)
{
    auto& mltk_micro_context = *mltk_tflite_micro_context(context);
    mltk_micro_context.layer_callback = callback;
    mltk_micro_context.layer_callback_arg = arg;
}


TfLiteStatus TfliteMicroModelHelper::invoke_layer_callback(
    TfLiteContext* context,
    const tflite::NodeAndRegistration& node_and_registration,
    TfLiteStatus invoke_status
)
{
    auto& mltk_micro_context = *mltk_tflite_micro_context(context);

    if(mltk_micro_context.layer_callback != nullptr)
    {
        return mltk_micro_context.layer_callback(
            mltk_micro_context.current_layer_index, 
            *context, 
            node_and_registration, 
            invoke_status, 
            mltk_micro_context.layer_callback_arg
        );
    }
    return invoke_status;
}


void TfliteMicroModelHelper::set_processing_callback(
    TfLiteContext* context,
    TfliteMicroProcessingCallback callback, 
    void *arg
)
{
    auto& mltk_micro_context = *mltk_tflite_micro_context(context);
    mltk_micro_context.processing_callback = callback;
    mltk_micro_context.processing_callback_arg = arg;
}


void TfliteMicroModelHelper::invoke_processing_callback(TfLiteContext* context)
{
    auto& mltk_micro_context = *mltk_tflite_micro_context(context);

    if(mltk_micro_context.processing_callback != nullptr)
    {
        mltk_micro_context.processing_callback(mltk_micro_context.processing_callback_arg);
    }
}

static TfLiteContext* _active_tflite_context = nullptr;
TfLiteContext* TfliteMicroModelHelper::active_tflite_context()
{
    return _active_tflite_context;
}

void TfliteMicroModelHelper::set_active_tflite_context(TfLiteContext* context)
{
    _active_tflite_context = context;
}

const char* TfliteMicroModelHelper::opcode_to_str(tflite::BuiltinOperator opcode)
{
  if (opcode == tflite::BuiltinOperator_CUSTOM) 
  {
    return "custom";
  } 
  else 
  {
    return tflite::EnumNameBuiltinOperator(opcode);
  }
}


const char* TfliteMicroModelHelper::create_layer_name(int layer_idx, tflite::BuiltinOperator opcode)
{
  static char op_name_buffer[128];
  snprintf(op_name_buffer, sizeof(op_name_buffer), "Op%d-%s", layer_idx, opcode_to_str(opcode));
  return op_name_buffer;
}


bool TfliteMicroModelHelper::verify_model_flatbuffer(const void* flatbuffer, int flatbuffer_length)
{
    flatbuffers::Verifier verifier((const uint8_t*)flatbuffer, flatbuffer_length);
    return tflite::VerifyModelBuffer(verifier);
}

const void* TfliteMicroModelHelper::get_metadata_from_tflite_flatbuffer(
    TfLiteContext* context, 
    const char* tag, 
    uint32_t* length
)
{
    return get_metadata_from_tflite_flatbuffer(
        TfliteMicroModelHelper::model(context), 
        tag, 
        length
    );
}

const void* TfliteMicroModelHelper::get_metadata_from_tflite_flatbuffer(
    const void* tflite_flatbuffer, 
    const char* tag, 
    uint32_t* length
)
{
    if(tflite_flatbuffer == nullptr)
    {
        return nullptr;
    }

    return get_metadata_from_tflite_flatbuffer(
        tflite::GetModel(tflite_flatbuffer), 
        tag, 
        length
    );
}


const void* TfliteMicroModelHelper::get_metadata_from_tflite_flatbuffer(
    const tflite::Model *model, 
    const char* tag, 
    uint32_t* length 
)
{
    if(length != nullptr)
    {
        *length = 0;
    }

    if(model == nullptr)
    {
        return nullptr;
    }

    const void* metadata_buffer = nullptr;

    const auto metadata_vector = model->metadata();
    if(metadata_vector == nullptr)
    {
        return nullptr;
    }

    const auto buffers_vector = model->buffers();
    if(buffers_vector == nullptr)
    {
        return nullptr;
    }

    for(auto meta : *metadata_vector)
    {
        if(meta == nullptr || meta->name() == nullptr)
        {
            return nullptr;
        }

        if(strcmp(meta->name()->c_str(), tag) == 0)
        {
            auto buffer_index = meta->buffer();
            auto buffer = buffers_vector->Get(buffer_index);
            if(buffer == nullptr)
            {
                return nullptr;
            }

            const auto buffer_data = buffer->data();
            if(buffer_data  == nullptr)
            {
                return nullptr;
            }

            metadata_buffer = buffer_data->Data();
            if(length != nullptr)
            {
                *length = buffer_data->size();
            }

            break;
        }
    }

    return metadata_buffer;
}


bool TfliteMicroModelHelper::get_tflite_flatbuffer_from_end_of_flash(
    const uint8_t** flatbuffer, 
    uint32_t* length, 
    const uint32_t* flash_end_addr
)
{
    *flatbuffer = nullptr;
    if(length != nullptr)
    {
        *length = 0;
    }

#if defined(FLASH_BASE) && defined(FLASH_SIZE)
    flash_end_addr = (flash_end_addr==nullptr) ? (const uint32_t*)(FLASH_BASE + FLASH_SIZE) : flash_end_addr;

    const uint32_t tflite_length = *(flash_end_addr-1);
    if(tflite_length == 0 || tflite_length >= FLASH_SIZE)
    {
        return false;
    }

    const uint8_t* tflite_flatbuffer = (const uint8_t*)flash_end_addr - sizeof(uint32_t) - tflite_length;
    if(verify_model_flatbuffer(tflite_flatbuffer, tflite_length))
    {
        *flatbuffer = tflite_flatbuffer;
        if(length != nullptr)
        {
            *length = tflite_length;
        }
        MLTK_INFO("Using .tflite in flash at 0x%08X", tflite_flatbuffer);
        return true;
    }
#endif

    return false;
}


} // namespace mltk 


#endif // MLTK_DLL_IMPORT