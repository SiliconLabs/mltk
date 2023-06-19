#pragma once

#include <new>
#include <cassert>
#include <vector>

#ifdef TFLITE_MICRO_ACCELERATOR_RECORDER_ENABLED
#include "mltk_tflite_micro_recorder.hpp"


#define TFLITE_MICRO_ACCELERATOR_RECORDER_CLEAR() ::mltk::TfliteMicroAcceleratorRecorder::instance().clear()
#define TFLITE_MICRO_ACCELERATOR_RECORDER_START_LAYER() ::mltk::TfliteMicroAcceleratorRecorder::instance().start_layer()
#define TFLITE_MICRO_ACCELERATOR_RECORDER_END_LAYER() ::mltk::TfliteMicroAcceleratorRecorder::instance().end_layer()
#define TFLITE_MICRO_ACCELERATOR_RECORD_PROGRAM(data, length) ::mltk::TfliteMicroAcceleratorRecorder::instance().record_program((const void*)data, length)
#define TFLITE_MICRO_ACCELERATOR_RECORD_PROGRAM_METADATA(key, data) ::mltk::TfliteMicroAcceleratorRecorder::instance().record_program_metadata(key, data)
#define TFLITE_MICRO_ACCELERATOR_RECORD_DATA(key, data, ...) ::mltk::TfliteMicroAcceleratorRecorder::instance().record_data(key, data, ##__VA_ARGS__)


namespace mltk
{





struct TfliteMicroAcceleratorRecorder
{
    bool program_recording_enabled = false;
    bool data_recording_enabled = false;
    bool layer_started = false;
    bool program_started = false;
    msgpack_context_t *program_context = nullptr;

    void clear()
    {
        msgpack_buffered_writer_deinit(program_context, true);
        program_context = nullptr;
        program_started = false;
    }

    void set_program_recording_enabled(bool enabled = true)
    {
        program_recording_enabled = enabled;
    }

    void set_data_recording_enabled(bool enabled = true)
    {
        data_recording_enabled = enabled;
    }

    void start_layer()
    {
        layer_started = true;
        program_started = false;
        if(program_recording_enabled)
        {
            msgpack_buffered_writer_init(&program_context, 4096);
            msgpack_write_array_marker(program_context, -1);
        }
    }

    void end_layer()
    {
        if(layer_started)
        {
            layer_started = false;

            if(program_context != nullptr)
            {
                if(msgpack_finalize_dynamic(program_context) == 0) // finalize the program array
                {
                    auto msgpack = get_layer_recording_context(true);
                    msgpack_write_dict_context(msgpack, "programs", program_context);
                }

                msgpack_buffered_writer_deinit(program_context, true);
                program_context = nullptr;
            }
        }
    }

    bool record_program(const void* data, unsigned length)
    {
        if(!program_recording_enabled || ! layer_started || program_context == nullptr)
        {
            return false;
        }

        _start_program();
        msgpack_write_dict_bin(program_context, "data", data, length);
        msgpack_finalize_dynamic(program_context); // finalize the program dict
        program_started = false;

        return true;
    }

    template<typename T>
    bool record_program_metadata(const char* key, T value)
    {
        if(!program_recording_enabled || ! layer_started || program_context == nullptr)
        {
            return false;
        }
        _start_program();
        msgpack_write_dict(program_context, key, value);

        return true;
    }

    template<typename T>
    bool record_program_metadata(const char* key, const std::vector<T>& data)
    {
        if(!program_recording_enabled || ! layer_started || program_context == nullptr)
        {
            return false;
        }

        _start_program();
        msgpack_write_dict_array(program_context, key, data.size());
        for(T e : data)
        {
            msgpack_write(program_context, e);
        }

        return true;
    }

    template<typename T>
    bool record_data(const char* key, const std::vector<T>& data)
    {
        if(!data_recording_enabled || !layer_started)
        {
            return false;
        }
        auto msgpack = get_layer_recording_context(true);
        msgpack_write_dict_array(msgpack, key, data.size());
        for(T e : data)
        {
            msgpack_write(msgpack, e);
        }
    }

    bool record_data(const char* key, const void* data, uint32_t length)
    {
        if(!data_recording_enabled || !layer_started)
        {
            return false;
        }
        auto msgpack = get_layer_recording_context(true);
        msgpack_write_dict_bin(msgpack, key, data, length);
    }

    bool record_data(const char* key, const uint8_t* data, uint32_t length)
    {
        if(!data_recording_enabled || !layer_started)
        {
            return false;
        }
        auto msgpack = get_layer_recording_context(true);
        msgpack_write_dict_bin(msgpack, key, data, length);
    }


    template<typename T>
    bool record_data(const char* key, const T* data, uint32_t length)
    {
        if(!data_recording_enabled || !layer_started)
        {
            return false;
        }
        auto msgpack = get_layer_recording_context(true);
        msgpack_write_dict_array(msgpack, key, length);
        for(int i = 0; i < length; ++i)
        {
            msgpack_write(msgpack, data[i]);
        }
    }

    template<typename T>
    bool record_data(const char* key, T value)
    {
        if(!data_recording_enabled || !layer_started)
        {
            return false;
        }
        auto msgpack = get_layer_recording_context(true);
        msgpack_write_dict(msgpack, key, value);
        return true;
    }


    static TfliteMicroAcceleratorRecorder& instance()
    {
        static TfliteMicroAcceleratorRecorder* instance = nullptr;
        static uint8_t instance_buffer[sizeof(TfliteMicroAcceleratorRecorder)];

        if(instance == nullptr)
        {
            instance = new(instance_buffer)TfliteMicroAcceleratorRecorder();
        }

        return *instance;
    }

private:
    void _start_program()
    {
        if(!program_started)
        {
            program_started = true;
            msgpack_write_dict_marker(program_context, -1);
        }
    }
};


} // namespace mltk


#else
#define TFLITE_MICRO_ACCELERATOR_RECORDER_CLEAR()
#define TFLITE_MICRO_ACCELERATOR_RECORDER_START_LAYER()
#define TFLITE_MICRO_ACCELERATOR_RECORDER_END_LAYER()
#define TFLITE_MICRO_ACCELERATOR_RECORD_PROGRAM(data, length)
#define TFLITE_MICRO_ACCELERATOR_RECORD_PROGRAM_METADATA(key, data)
#define TFLITE_MICRO_ACCELERATOR_RECORD_DATA(key, data, ...)

#endif

