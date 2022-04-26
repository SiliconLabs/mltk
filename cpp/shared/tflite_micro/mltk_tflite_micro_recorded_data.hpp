#pragma once 

#include "tensorflow/lite/c/common.h"
#include "cpputils/typed_list.hpp"


#ifdef TFLITE_MICRO_RECORDER_ENABLED
#define TFLITE_MICRO_RECORDER_CLEAR() TfliteMicroRecordedData::instance().clear()
#else 
#define TFLITE_MICRO_RECORDER_CLEAR()
#endif



namespace mltk
{

struct TfliteMicroRecordedTensor
{
    uint8_t* data = nullptr;
    unsigned length = 0;

    TfliteMicroRecordedTensor(const TfLiteTensor* tensor);
    void clear();
};


struct TfliteMicroRecordedLayer
{
    cpputils::TypedList<TfliteMicroRecordedTensor> inputs;
    cpputils::TypedList<TfliteMicroRecordedTensor> outputs;

    void clear();
};

struct TfliteMicroRecordedData : public cpputils::TypedList<TfliteMicroRecordedLayer>
{
    ~TfliteMicroRecordedData();
    void clear(void);
    static TfliteMicroRecordedData& instance();
};


} // namespace mltk