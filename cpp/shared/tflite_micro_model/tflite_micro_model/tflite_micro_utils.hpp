#pragma once

#include <cstdint>
#include <cmath>


namespace mltk
{




/**
 * @brief Scale the source buffer by the given scaler
 * 
 * dst_float32 = src * scaler
 */
template<typename SrcType>
void scale_tensor(float scaler, const SrcType* src, float* dst, uint32_t length)
{
    for(; length > 0; --length)
    {
        const float src_flt = static_cast<SrcType>(*src++);
        *dst++ = src_flt * scaler;
    }
}

/**
 * @brief Normalize the source buffer by mean and STD
 * 
 * dst_float32 = (src - mean(src)) / std(src)
 */
template<typename SrcType>
void samplewise_mean_std_tensor(const SrcType* src, float* dst, uint32_t length)
{
    float mean = 0.0f;
    float count = 0.0f;
    float m2 = 0.0f;
    const SrcType *ptr;

    // Calculate the STD and mean
    ptr = src;
    for(int i = length; i > 0; --i)
    {
        const float value = (float)(*ptr++);

        count += 1;

        const float delta = value - mean;
        mean += delta / count;
        const float delta2 = value - mean;
        m2 += delta * delta2;
    }

    const float variance = m2 / count;
    const float std = sqrtf(variance);
    const float std_recip = 1.0f / std; // multiplication is faster than division

    // Subtract the mean and divide by the STD
    ptr = src;
    for(int i = length; i > 0; --i)
    {
        const float value = (float)(*ptr++);
        const float x = value - mean;

        *dst++ = x * std_recip;
    }
}



} // namespace mltk