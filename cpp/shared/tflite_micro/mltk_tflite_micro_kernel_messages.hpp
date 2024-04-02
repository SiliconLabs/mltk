#pragma once 

#include <string>

#include "cpputils/helpers.hpp"
#include "logging/logging.hpp"




namespace mltk 
{


#define MLTK_KERNEL_UNSUPPORTED_MSG(fmt, ...) ::mltk::TfliteMicroKernelMessages::issue(fmt, ## __VA_ARGS__);


class DLL_EXPORT TfliteMicroKernelMessages
{
public:
    using FlushCallback = void (*)(const char* msg, void* arg) ;

    static TfliteMicroKernelMessages& instance();
    static void issue(const char* fmt, ...);
    static void flush(logging::Level level = logging::Warn);
    static void reset();
    static bool have_messages();
    static void set_enabled(bool enabled);

    static bool unknown_layers_detected();
    static void set_unknown_layers_detected(bool detected);
    static void set_flush_callback(FlushCallback callback, void *arg = nullptr);

private:
    std::string _unsupported_msg;
    FlushCallback _flush_callback = nullptr;
    void* _flush_callback_arg = nullptr;
    bool _enabled = true;
    bool _unknown_layers_detected = false;

    TfliteMicroKernelMessages() = default;
};


} // namespace mltk 