#pragma once 

#include <vector>

#include "cpputils/helpers.hpp"
#include "mltk_tflite_micro_recorder.hpp"



namespace mltk
{

class DLL_EXPORT TfliteMicroAcceleratorRecorder
{
public:
    static TfliteMicroAcceleratorRecorder& instance();
    static bool is_enabled();
    static void set_enabled(bool enabled);
    static bool is_program_recording_enabled();
    static void set_program_recording_enabled(bool enabled);
    static msgpack_context_t* programs_context();

    static void reset();
    static bool record_data(const char* key, const void* data, uint32_t length);
    static bool record_data(const char* key, const uint8_t* data, uint32_t length);
    static bool record_program(const void* data, unsigned length);
    
    template<typename T>
    static bool record_program_metadata(const char* key, T value)
    {
        auto& self = instance();

        if(self._programs_context == nullptr)
        {
            return false;
        }

        _start_program();
        msgpack_write_dict(self._programs_context, key, value);

        return true;
    }

    template<typename T>
    static bool record_program_metadata(
        const char* key,
        const std::vector<T>& data,
        void (*write_callback)(msgpack_context_t*, const void*) = nullptr
    )
    {
        auto& self = instance();

        if(self._programs_context == nullptr)
        {
            return false;
        }

        _start_program();
        msgpack_write_dict_array(self._programs_context, key, data.size());
        for(const T& e : data)
        {
            if(write_callback != nullptr)
            {
                write_callback(self._programs_context, &e);
            }
            else
            {
                msgpack_write(self._programs_context, e);
            }
        }

        return true;
    }

    template<typename T>
    static bool record_data(const char* key, const T* data, uint32_t length)
    {
        auto& self = instance();

        if(!self._enabled)
        {
            return false;
        }
        auto msgpack = TfliteMicroRecorder::get_context();
        msgpack_write_dict_array(msgpack, key, length);
        for(int i = 0; i < length; ++i)
        {
            msgpack_write(msgpack, data[i]);
        }
    }

    template<typename T>
    static bool record_data(const char* key, T value)
    {
        auto& self = instance();

        if(!self._enabled)
        {
            return false;
        }
        auto msgpack = TfliteMicroRecorder::get_context();
        msgpack_write_dict(msgpack, key, value);
        return true;
    }

    template<typename T>
    static bool record_data(const char* key, const std::vector<T>& data)
    {
        auto& self = instance();

        if(!self._enabled)
        {
            return false;
        }
        auto msgpack = TfliteMicroRecorder::get_context();

        msgpack_write_dict_array(msgpack, key, data.size());
        for(T e : data)
        {
            msgpack_write(msgpack, e);
        }

        return true;
    }


private:
    msgpack_context_t *_programs_context = nullptr;
    bool _program_active = false;
    bool _enabled = false; 
    bool _program_recording_enabled = false;

    static void _on_layer_execution_started();
    static void _on_layer_execution_ending();
    static void _start_program();

    TfliteMicroAcceleratorRecorder() = default;
};




} // namespace mltk



#ifdef TFLITE_MICRO_ACCELERATOR_RECORDER_ENABLED

#define MLTK_ACCELERATOR_RECORDER_RESET() ::mltk::TfliteMicroAcceleratorRecorder::reset()
#define MLTK_ACCELERATOR_RECORD_PROGRAM(data, length) ::mltk::TfliteMicroAcceleratorRecorder::record_program(data, length)
#define MLTK_ACCELERATOR_RECORD_PROGRAM_METADATA(key, data, ...) ::mltk::TfliteMicroAcceleratorRecorder::record_program_metadata(key, data, ## __VA_ARGS__)
#define MLTK_ACCELERATOR_RECORD_DATA(key, data, ...) ::mltk::TfliteMicroAcceleratorRecorder::record_data(key, data, ## __VA_ARGS__)

#else 

#define MLTK_ACCELERATOR_RECORDER_RESET()
#define MLTK_ACCELERATOR_RECORD_PROGRAM(data, length)
#define MLTK_ACCELERATOR_RECORD_PROGRAM_METADATA(key, data, ...)
#define MLTK_ACCELERATOR_RECORD_DATA(key, data, ...)

#endif // TFLITE_MICRO_ACCELERATOR_RECORDER_ENABLED