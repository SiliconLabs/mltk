#pragma once 

#include <complex>
#include <algorithm>
#include <cassert>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "float16.hpp"


namespace mltk 
{

typedef std::complex<float> cfloat_t;
typedef std::complex<float16_t> cfloat16_t;



/*************************************************************************************************/
inline tflite::PaddingType runtime_padding_type(TfLitePadding padding)
{
    switch (padding)
    {
    case TfLitePadding::kTfLitePaddingSame:
        return tflite::PaddingType::kSame;
    case TfLitePadding::kTfLitePaddingValid:
        return tflite::PaddingType::kValid;
    case TfLitePadding::kTfLitePaddingUnknown:
    default:
        return tflite::PaddingType::kNone;
    }
}

/*************************************************************************************************/
inline int div_round_up(const int numerator, const int denominator)
{
    return (numerator + denominator - 1) / denominator;
}

/*************************************************************************************************/
inline int div_ceil(const int numerator, const int denominator, int lower_limit = std::numeric_limits<int>::min())
{
    assert(denominator != 0);
    const std::div_t div = std::div(numerator, denominator);

    auto sign = [](int n){ return n > 0 ? 1 : n < 0 ? -1 : 0; };
    int ceil;
    if (sign(numerator) != sign(denominator)) {
        ceil = div.quot;  // negative number already truncated towards 0
    } else { // exact 0 or positive number truncated toward 0
        ceil = div.quot + (div.rem ? 1 : 0);
    }

    return std::max(ceil, lower_limit);
}

/*************************************************************************************************/
inline int div_floor(const int numerator, const int denominator, int upper_limit = std::numeric_limits<int>::max())
{
    assert(denominator != 0);
    const std::div_t div = std::div(numerator, denominator);

    auto sign = [](int n){ return n > 0 ? 1 : n < 0 ? -1 : 0; };
    int floor;
    if (sign(numerator) == sign(denominator)) {
        floor = div.quot;  // positive number already truncated towards 0
    } else { // exact 0 or negative number truncated toward 0
        floor = div.quot - (div.rem ? 1 : 0);
    }

    return std::min(floor, upper_limit);
}


} // namespace mltk