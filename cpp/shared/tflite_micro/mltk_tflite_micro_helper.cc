#include <cstdarg>
#include <cassert>
#include <string>
#include "em_device.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "mltk_tflite_micro_internal.hpp"
#include "cpputils/std_formatted_string.hpp"

namespace mltk
{

static std::string _unsupported_msg;
static Logger *mltk_logger =  nullptr;
bool model_profiler_enabled = false;
static bool _unsupported_kernel_messages_enabled = true;

#ifdef TFLITE_MICRO_VERSION_STR
const char* TFLITE_MICRO_VERSION = TFLITE_MICRO_VERSION_STR;
#else
const char* TFLITE_MICRO_VERSION = nullptr;
#endif


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
    if(_current_kernel_index == -1)
    {
        return;
    }

    if(has_unsupported_kernel_messages())
    {
        va_list args;
        va_start(args, fmt);
        _unsupported_msg = _unsupported_msg + ", " + cpputils::vformat(fmt, args);
        va_end(args);
    }
    else
    {
        va_list args;
        va_start(args, fmt);
        _unsupported_msg = cpputils::vformat(fmt, args);
        va_end(args);
    }
}

/*************************************************************************************************/
extern "C" void flush_unsupported_kernel_messages(logging::Level level)
{
    if(has_unsupported_kernel_messages())
    {
        if(_unsupported_kernel_messages_enabled)
        {
            auto& logger = get_logger();
            logger.write(level, "%s not supported: %s", get_current_layer_str(), _unsupported_msg.c_str());
        }
        reset_unsupported_kernel_messages();
    }
}

/*************************************************************************************************/
extern "C" void reset_unsupported_kernel_messages()
{
    _unsupported_msg.clear();
}

/*************************************************************************************************/
extern "C" bool has_unsupported_kernel_messages()
{
    return _unsupported_msg.size() > 0;
}

/*************************************************************************************************/
extern "C" void set_unsupported_kernel_messages_enabled(bool enabled)
{
    _unsupported_kernel_messages_enabled = enabled;
}

/*************************************************************************************************/
extern "C" void mltk_tflite_micro_get_current_layer_opcode_and_index(int* opcode, int* index)
{
    *opcode = mltk::_current_kernel_op_code;
    *index = mltk::_current_kernel_index;
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
bool get_tflite_flatbuffer_from_end_of_flash(const uint8_t** flatbuffer, uint32_t* length, const uint32_t* flash_end_addr)
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

/*************************************************************************************************/
bool verify_model_flatbuffer(const void* flatbuffer, int flatbuffer_length)
{
    flatbuffers::Verifier verifier((const uint8_t*)flatbuffer, flatbuffer_length);
    return tflite::VerifyModelBuffer(verifier);
}

/*************************************************************************************************/
const char* to_str(tflite::BuiltinOperator op_type)
{
  if (op_type == tflite::BuiltinOperator_CUSTOM) {
    return "custom";
  } else {
    return tflite::EnumNameBuiltinOperator(op_type);
  }
}

/*************************************************************************************************/
const char* op_to_str(int op_idx, tflite::BuiltinOperator op_type)
{
  static char op_name_buffer[128];
  snprintf(op_name_buffer, sizeof(op_name_buffer), "Op%d-%s", op_idx, to_str(op_type));
  return op_name_buffer;
}




} // namespace mltk
