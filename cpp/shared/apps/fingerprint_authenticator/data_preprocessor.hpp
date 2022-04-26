#pragma once 

#include <cstdint>

#include "tflite_model_parameters/tflite_model_parameters.hpp"

namespace mltk 
{

class DataPreprocessor
{
public:
    DataPreprocessor() = default;

    bool load(const TfliteModelParameters& params, uint16_t width, uint16_t height);

    bool verify_sample(const uint8_t* sample) const;
    bool preprocess_sample(const uint8_t* unprocessed_sample, uint8_t* processed_sample) const;
    void crop_image(const uint8_t* src, uint8_t* dst) const;
    void balance_colorspace(uint8_t* sample) const;
    void sharpen_image(uint8_t* sample) const;

private:
    uint16_t _width;
    uint16_t _height;
    const int8_t* _sharpen_filter;
    uint8_t _sharpen_filter_width;
    uint8_t _sharpen_filter_height;
    uint8_t _sharpen_gain;
    uint8_t _balance_threshold_max;
    uint8_t _balance_threshold_min;
    uint8_t _border;
    uint8_t _verify_imin;
    uint8_t _verify_imax;
    uint16_t _verify_full_threshold;
    uint16_t _verify_center_threshold;
};

} // namespace mltk 