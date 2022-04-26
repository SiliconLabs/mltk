#pragma once 

#include <new>
#include <cassert>
#include "tensorflow/lite/c/common.h"
#include "cpputils/typed_list.hpp"


#ifdef TFLITE_MICRO_ACCELERATOR_RECORDER_ENABLED
#define TFLITE_MICRO_ACCELERATOR_RECORDER_CLEAR() ::mltk::TfliteMicroAcceleratorRecorder::instance().clear()
#define TFLITE_MICRO_ACCELERATOR_RECORDER_START_LAYER() ::mltk::TfliteMicroAcceleratorRecorder::instance().start_layer()
#define TFLITE_MICRO_ACCELERATOR_RECORDER_END_LAYER() ::mltk::TfliteMicroAcceleratorRecorder::instance().end_layer()
#define TFLITE_MICRO_ACCELERATOR_RECORDER_ADD(name, data, length) ::mltk::TfliteMicroAcceleratorRecorder::instance().add_layer_data(name, (const void*)data, length)
#else 
#define TFLITE_MICRO_ACCELERATOR_RECORDER_CLEAR()
#define TFLITE_MICRO_ACCELERATOR_RECORDER_START_LAYER()
#define TFLITE_MICRO_ACCELERATOR_RECORDER_END_LAYER()
#define TFLITE_MICRO_ACCELERATOR_RECORDER_ADD(name, data, length)
#endif



namespace mltk
{

struct TfliteMicroAcceleratorRecordedBuffer
{
    uint8_t* data = nullptr;
    unsigned length = 0;

    TfliteMicroAcceleratorRecordedBuffer(const void* data, unsigned length)
    {
        this->data = (uint8_t*)malloc(length);
        assert(this->data != nullptr);
        memcpy(this->data, data, length);
        this->length = length;
    }
    void clear()
    {
        if(this->data != nullptr)
        {
            free(this->data);
            this->data = nullptr;
            this->length = 0;
        }
    }
};


struct TfliteMicroAcceleratorRecordedBufferList : public cpputils::TypedList<TfliteMicroAcceleratorRecordedBuffer>
{
    bool add(const void* data, unsigned length)
    {
        TfliteMicroAcceleratorRecordedBuffer buffer(data, length);
        return this->append(buffer);
    }

    void clear()
    {
        for(auto& value : *this)
        {
            value.clear();
        }
        cpputils::TypedList<TfliteMicroAcceleratorRecordedBuffer>::clear();
    }  
};


struct TfliteMicroAcceleratorRecordedLayer : public cpputils::TypedDict<TfliteMicroAcceleratorRecordedBufferList>
{
    void clear()
    {
        for(auto value : *this)
        {
            value->clear();
        }
        cpputils::TypedDict<TfliteMicroAcceleratorRecordedBufferList>::clear();
    }
};

struct TfliteMicroAcceleratorRecorder : public cpputils::TypedList<TfliteMicroAcceleratorRecordedLayer>
{
    bool enabled = false;
    TfliteMicroAcceleratorRecordedLayer* current_layer = nullptr;

    ~TfliteMicroAcceleratorRecorder()
    {
        clear();
    }

    void clear(void)
    {
        for(auto& e : *this)
        {
            e.clear();
        }
        cpputils::TypedList<TfliteMicroAcceleratorRecordedLayer>::clear();
        enabled = false;
    }

    void set_enabled()
    {
        clear();
        enabled = true;
    }

    void start_layer()
    {
        if(enabled)
        {
            TfliteMicroAcceleratorRecordedLayer layer;
            this->append(layer);
            current_layer = &this->last();
        }
    }

    void end_layer()
    {
        current_layer = nullptr;
    }

    bool add_layer_data(const char* name, const void* data, unsigned length)
    {
        if(!enabled)
        {
            return false;
        }

        assert(current_layer != nullptr);

        if(!current_layer->contains(name))
        {
            TfliteMicroAcceleratorRecordedBufferList buffer_list;
            current_layer->put(name, &buffer_list);
        }

        auto buffer_list = current_layer->get(name);
        return buffer_list->add(data, length);
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