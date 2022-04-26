#pragma once

#include <cstdint>
#include <cmath>

#include "cpputils/string.hpp"
#include "logging/logger.hpp"
#include "tflite_micro_model/tflite_micro_model.hpp"
#include "tflite_model_parameters/tflite_model_parameters.hpp"


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



static inline uint32_t get_tensor_arena_size(const void* flatbuffer, logging::Logger* logger = nullptr)
{
    uint32_t runtime_memory_size;
    TfliteModelParameters model_parameters;
    
    // Attempt to retrieve the runtime memory size from the model parameters
    if(TfliteModelParameters::load_from_tflite_flatbuffer(flatbuffer, model_parameters))
    {
        if(model_parameters.get("runtime_memory_size", runtime_memory_size))
        {
            if(logger != nullptr)
            {
                logger->info("runtime_memory_size from .tflite is %s", cpputils::format_units(runtime_memory_size));
            }
            return runtime_memory_size;
        }
    }

    // Otherwise just default to a large value
    runtime_memory_size = 184*1024; // 184k for embedded
    if(logger != nullptr)
    {
        logger->info("No runtime_memory_size found in .tflite, defaulting to %s", cpputils::format_units(runtime_memory_size));
    }

    return runtime_memory_size;
}


} // namespace mltk