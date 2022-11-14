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
#define TFLITE_MICRO_ACCELERATOR_RECORD_DATA(key, data, ...) ::mltk::TfliteMicroAcceleratorRecorder::instance().record_data(key, data, ##__VA_ARGS__)


namespace mltk
{


struct RecordedProgram 
{
    void* data;
    uint32_t length;

    RecordedProgram() = default;

    RecordedProgram(const void* data, uint32_t length)
    {
        this->data = malloc(length);
        if(this->data != nullptr)
        {
            this->length = length;
            memcpy(this->data, data, length);
        }
        else 
        {
            this->length = 0;
        }
    }

    RecordedProgram(RecordedProgram && rhs)
    {
        data = rhs.data;
        length = rhs.length;
        rhs.data = nullptr;
        rhs.length = 0;
    }

    RecordedProgram& operator=(RecordedProgram && rhs)
    {
        this->data = rhs.data;
        this->length = rhs.length;

        rhs.data = nullptr;
        rhs.length = 0;
        return *this;
    }

    ~RecordedProgram()
    {
        if(this->data != nullptr)
        {
            free(this->data);
        }
        this->data = nullptr;
        this->length = 0;
    }
};



struct TfliteMicroAcceleratorRecorder
{
    bool program_recording_enabled = false;
    bool data_recording_enabled = false;
    bool layer_started = false;
    std::vector<RecordedProgram> programs;

    void clear() 
    {
        programs.clear();
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
    }

    void end_layer()
    {
        if(layer_started)
        {
            layer_started = false;
            auto msgpack = get_layer_recording_context(true);

            if(programs.size() > 0)
            {
                msgpack_write_dict_array(msgpack, "programs", programs.size());
                for(auto& p : programs)
                {
                    msgpack_write_bin(msgpack, p.data, p.length);
                }
                programs.clear();
            }
        }
    }

    bool record_program(const void* data, unsigned length)
    {
        if(!program_recording_enabled || ! layer_started)
        {
            return false;
        }
        programs.push_back({data, length});

        return true;
    }

    template<typename T>
    bool record_data(const char* key, const std::vector<T> data)
    {
        if(!data_recording_enabled || !layer_started)
        {
            return false;
        }
        auto msgpack = get_layer_recording_context(true);
        msgpack_write_dict_array(msgpack, key, data.size());
        for(auto& e : data)
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
};


} // namespace mltk


#else 
#define TFLITE_MICRO_ACCELERATOR_RECORDER_CLEAR()
#define TFLITE_MICRO_ACCELERATOR_RECORDER_START_LAYER()
#define TFLITE_MICRO_ACCELERATOR_RECORDER_END_LAYER()
#define TFLITE_MICRO_ACCELERATOR_RECORD_PROGRAM(data, length)
#define TFLITE_MICRO_ACCELERATOR_RECORD_DATA(key, data, ...)

#endif

