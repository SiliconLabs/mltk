#include <cassert>
#include <cstdio>

#include "tflite_micro_model/tflite_micro_tensor.hpp"

namespace mltk 
{


/*************************************************************************************************/
 TfliteTensorShape TfliteTensorView::shape() const
 {
     return TfliteTensorShape(this->dims);
 }

/*************************************************************************************************/
unsigned TfliteTensorView::element_size() const
{
    switch(this->type)
    {
    case kTfLiteBool:
    case kTfLiteInt8:
    case kTfLiteUInt8:
        return 1;
    case kTfLiteInt16:
        return 2;
    case kTfLiteInt32:
    case kTfLiteFloat32:   
        return 4;
    case kTfLiteFloat64:   
    case kTfLiteUInt64:
    case kTfLiteInt64:
        return 8;
    default: 
        return 0;
    }
}

/*************************************************************************************************/
const char* TfliteTensorView::to_str(char* str_buffer) const
{
    static char local_buffer[64];
    char shape_str[32];
    str_buffer = (str_buffer == nullptr) ? local_buffer : str_buffer;

    snprintf(str_buffer, 64, "%s (%s)", this->shape().to_str(shape_str), ::mltk::to_str(this->type));

    return str_buffer;
}


/*************************************************************************************************/
TfliteTensorShape::TfliteTensorShape(const TfLiteIntArray* dims)
{
    init(dims);
}

/*************************************************************************************************/
void TfliteTensorShape::init(const TfLiteIntArray* dims)
{
    this->length = dims->size;
    for(int i = 0; i < this->length; ++i)
    {
        this->dims[i] = dims->data[i];
    }
}

/*************************************************************************************************/
uint32_t TfliteTensorShape::flat_size() const
{
    uint32_t s = 1;
    for(int i = 0; i < length; ++i)
    {
        s *= dims[i];
    }
    return s;
}

/*************************************************************************************************/
uint32_t TfliteTensorShape::operator [](int i) const
{
    if(i >= length)
    {
        assert(!"Invalid index");
        return 0;
    }
    return dims[i];
}

/*************************************************************************************************/
char* TfliteTensorShape::to_str(char* str_buffer) const
{
    char* p = str_buffer;
    int i = 0;

    for(; i < length; ++i)
    {
        p += sprintf(p, "%dx", dims[i]);
    }

    if(i > 0)
    {
        *(p - 1) = 0;
    }
    else 
    {
        *p = 0;
    }

    return str_buffer;
}

/*************************************************************************************************/
const char* to_str(TfLiteType dtype)
{
    switch(dtype)
    {
    case kTfLiteBool:
        return "bool";
    case kTfLiteInt8:
        return "int8";
    case kTfLiteUInt8:
        return  "uint8";
    case kTfLiteInt16:
        return "int16";
    case kTfLiteInt32:
        return "int32";
    case kTfLiteFloat32:   
        return "float32";
    case kTfLiteFloat64:   
        return "float64";
    case kTfLiteUInt64:
        return "uint64";
    case kTfLiteInt64:
        return "int64";
    default: 
        return "unknown";
    }
}


} // namespace mltk