#ifndef MLTK_DLL_IMPORT

#include <new>

#include "cpputils/std_formatted_string.hpp"

#include "mltk_tflite_micro_kernel_messages.hpp"
#include "mltk_tflite_micro_model_helper.hpp"
#include "mltk_tflite_micro_logger.hpp"


namespace mltk 
{


TfliteMicroKernelMessages& TfliteMicroKernelMessages::instance()
{
    static uint8_t instance_buffer[sizeof(TfliteMicroKernelMessages)];
    static TfliteMicroKernelMessages* instance_ptr = nullptr;

    if(instance_ptr == nullptr)
    {
        instance_ptr = new(instance_buffer)TfliteMicroKernelMessages();
    }

    return *instance_ptr;
}


void TfliteMicroKernelMessages::issue(const char* fmt, ...)
{
    auto& self = instance();

    if(TfliteMicroModelHelper::current_layer_index() == -1)
    {
        return;
    }

    if(have_messages())
    {
        va_list args;
        va_start(args, fmt);
        self._unsupported_msg = self._unsupported_msg + ", " + cpputils::vformat(fmt, args);
        va_end(args);
    }
    else
    {
        va_list args;
        va_start(args, fmt);
        self._unsupported_msg = cpputils::vformat(fmt, args);
        va_end(args);
    }
}


void TfliteMicroKernelMessages::flush(logging::Level level)
{
    auto& self = instance();

    if(have_messages())
    {
        if(self._enabled)
        {
            auto& logger = get_logger();
            const auto &msg = cpputils::format(
                "%s not supported: %s",
                TfliteMicroModelHelper::current_layer_name(), 
                self._unsupported_msg.c_str()
            );
            logger.write(level, msg.c_str());
            if(self._flush_callback != nullptr)
            {
                self._flush_callback(msg.c_str(), self._flush_callback_arg);
            }
        }
        reset();
    }
}


void TfliteMicroKernelMessages::reset()
{
    auto& self = instance();
    self._unsupported_msg.clear();
}


bool TfliteMicroKernelMessages::have_messages()
{
    auto& self = instance();
    return self._unsupported_msg.size() > 0;
}


void TfliteMicroKernelMessages::set_enabled(bool enabled)
{
    auto& self = instance();
    self._enabled = true;
}

bool TfliteMicroKernelMessages::unknown_layers_detected()
{
    auto& self = instance();
    return self._unknown_layers_detected;
}

void TfliteMicroKernelMessages::set_unknown_layers_detected(bool detected)
{
    auto& self = instance();
    self._unknown_layers_detected = detected;
}

void TfliteMicroKernelMessages::set_flush_callback(FlushCallback callback, void* arg)
{
    auto& self = instance();
    self._flush_callback = callback;
    self._flush_callback_arg = arg;
}

} // namespace mltk 

#endif // MLTK_DLL_IMPORT