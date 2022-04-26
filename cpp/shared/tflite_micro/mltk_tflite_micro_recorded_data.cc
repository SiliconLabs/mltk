#if TFLITE_MICRO_RECORDER_ENABLED
#include <cstring>
#include <cassert>

#include "mltk_tflite_micro_recorded_data.hpp"


namespace mltk
{


static TfliteMicroRecordedData _recorded_data;


/*************************************************************************************************/
TfliteMicroRecordedData& TfliteMicroRecordedData::instance()
{
    return _recorded_data;
}

/*************************************************************************************************/
TfliteMicroRecordedTensor::TfliteMicroRecordedTensor(const TfLiteTensor* tensor)
{
    if(tensor == nullptr)
    {
        this->data = nullptr;
        this->length = 0;
    }
    else
    {
        this->data = (uint8_t*)malloc(tensor->bytes);
        assert(this->data != nullptr);
        memcpy(this->data, tensor->data.raw, tensor->bytes);
        this->length = tensor->bytes;
    }
}

/*************************************************************************************************/
void TfliteMicroRecordedTensor::clear()
{
    if(this->data != nullptr)
    {
        free(this->data);
        this->data = nullptr;
        this->length = 0;
    }
}

/*************************************************************************************************/
void TfliteMicroRecordedLayer::clear()
{
    for(auto& i : this->inputs)
    {
        i.clear();
    }
    this->inputs.clear();

    for(auto& i : this->outputs)
    {
        i.clear();
    }
    this->outputs.clear();
}

/*************************************************************************************************/
TfliteMicroRecordedData::~TfliteMicroRecordedData()
{
    clear();
}

/*************************************************************************************************/
void TfliteMicroRecordedData::clear(void)
{
    for(auto& i : *this)
    {
        i.clear();
    }
    cpputils::TypedList<TfliteMicroRecordedLayer>::clear();
}


} // namespace mltk


#endif // TFLITE_MICRO_RECORDER_ENABLED
