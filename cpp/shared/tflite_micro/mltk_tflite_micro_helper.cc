#include <cstdarg>
#include <cassert>
#include "em_device.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "mltk_tflite_micro_internal.hpp"


namespace mltk
{

static Logger *mltk_logger =  nullptr;
bool model_profiler_enabled = false;
bool model_recorder_enabled = false;


/*************************************************************************************************/
TfLiteStatus allocate_scratch_buffer(TfLiteContext *ctx, unsigned size_bytes, int *scratch_buffer_index)
{
    auto status = ctx->RequestScratchBufferInArena(ctx, size_bytes, scratch_buffer_index);
    if(status != kTfLiteOk)
    {
        assert(!"Failed to allocate scratch buffer");
    }
    return status; 
}

/*************************************************************************************************/
#ifndef MLTK_DLL_IMPORT 
extern "C" void issue_unsupported_kernel_message(const char* fmt, ...)
{
  if(_current_kernel_index == -1 || _issued_unsupported_msg)
  {
    return;
  }

  _issued_unsupported_msg = true;

  char buffer[256];
  char op_name[92];
  const int l = snprintf(buffer, sizeof(buffer), "%s not supported: ", op_to_str(_current_kernel_index, (tflite::BuiltinOperator)_current_kernel_op_code));

  va_list args;
  va_start(args, fmt);
  vsnprintf(&buffer[l], sizeof(buffer)-l, fmt, args);
  va_end(args);

  get_logger().warn("%s", buffer);
}
#endif // MLTK_DLL_IMPORT

/*************************************************************************************************/
Logger& get_logger()
{
    if(mltk_logger == nullptr)
    {
        mltk_logger = logging::get("MLTK");
        if(mltk_logger == nullptr)
        {
            mltk_logger = logging::create("MLTK", LogLevel::Info);
            assert(mltk_logger != nullptr);
        }
    }

    return *mltk_logger;
}

/*************************************************************************************************/
bool set_log_level(LogLevel level)
{
    return get_logger().level(level);
}

/*************************************************************************************************/
#ifndef MLTK_DLL_IMPORT 
static const TfliteMicroAccelerator* _registered_accelerator = nullptr;
extern "C" void mltk_tflite_micro_set_accelerator(const TfliteMicroAccelerator* accelerator)
{
    _registered_accelerator = accelerator;
}
#endif

/*************************************************************************************************/
#ifndef MLTK_DLL_IMPORT 
extern "C" const TfliteMicroAccelerator* mltk_tflite_micro_get_registered_accelerator()
{
    return _registered_accelerator;
}
#endif


#ifndef TFLITE_MICRO_ACCELERATOR
/*************************************************************************************************/
extern "C" void mltk_tflite_micro_register_accelerator()
{
    // This is just a placeholder if no accelerator is built into the binary / shared library
    // (i.e. each accelerator also defines this API and internally calls mltk_tflite_micro_set_accelerator())
    mltk_tflite_micro_set_accelerator(nullptr);
}
#endif



/*************************************************************************************************/
const void* get_metadata_from_tflite_flatbuffer(const void* tflite_flatbuffer, const char* tag, uint32_t* length)
{
    if(tflite_flatbuffer == nullptr)
    {
        return nullptr;
    }

    const void* metadata_buffer = nullptr;
    const auto model = tflite::GetModel(tflite_flatbuffer);
    if(model == nullptr)
    {
        return nullptr;
    }

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

/*************************************************************************************************/
int TfliteMicroErrorReporter::Report(const char* format, va_list args)
{
    auto& logger = get_logger();
    const auto orig_flags = logger.flags();
    logger.flags().clear(logging::Newline);
    logger.vwrite(logging::Error, format, args);
    logger.write(logging::Error, "\n");
    logger.flags(orig_flags);
    return 0;
}

/*************************************************************************************************/
bool get_tflite_flatbuffer_from_end_of_flash(const uint8_t** flatbuffer, uint32_t* length)
{
    *flatbuffer = nullptr;
    if(length != nullptr)
    {
        *length = 0;
    }

#ifdef __arm__
    const uint32_t *flash_end_addr = (const uint32_t*)(FLASH_BASE + FLASH_SIZE);

    const uint32_t tflite_length = *(flash_end_addr-1);
    if(tflite_length == 0 || tflite_length > 1024*1024)
    {
        return false;
    }

    const uint8_t* tflite_flatbuffer = (const uint8_t*)flash_end_addr - sizeof(uint32_t) - tflite_length;
    flatbuffers::Verifier verifier(tflite_flatbuffer, tflite_length);
    if(tflite::VerifyModelBuffer(verifier))
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
