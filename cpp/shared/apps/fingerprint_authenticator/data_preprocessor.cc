
#include <algorithm>
#include <cstring>

#include "fingerprint_reader/fingerprint_reader.h"
#include "jlink_stream/jlink_stream.hpp"
#include "mltk_tflite_micro_helper.hpp"

#include "data_preprocessor.hpp"



namespace mltk 
{


/*************************************************************************************************/
bool DataPreprocessor::load(const TfliteModelParameters& params, uint16_t width, uint16_t height)
{
    jlink_stream::register_stream("preprocessed", jlink_stream::Write);

    _width = width;
    _height = height;
    params.get("sharpen_filter", (const uint8_t*&)_sharpen_filter);
    params.get("sharpen_filter_width", _sharpen_filter_width);
    params.get("sharpen_filter_height", _sharpen_filter_height);
    params.get("sharpen_gain", _sharpen_gain);
    params.get("balance_threshold_max", _balance_threshold_max);
    params.get("balance_threshold_min", _balance_threshold_min);
    params.get("border", _border);
    params.get("verify_imin", _verify_imin);
    params.get("verify_imax", _verify_imax);
    params.get("verify_full_threshold", _verify_full_threshold);
    params.get("verify_center_threshold", _verify_center_threshold);

    return true;
}


/*************************************************************************************************/
bool DataPreprocessor::verify_sample(const uint8_t* sample) const
{
    const uint8_t* p;
    int32_t dark_full = 0;
    int32_t light_full = 0;

    p = sample;
    for(int i = 0; i < _height; ++i)
    {
        for(int j = 0; j < _width; ++j)
        {
            if(*p < _verify_imin)
            {
                ++dark_full;
            }
            else if(*p >= _verify_imax)
            {
                ++light_full;
            }

            ++p;
        }
    }

    int32_t dark_center = 0;
    int32_t light_center = 0;

    p = sample + _border*_width;
    for(int i = _border; i < _height - _border; ++i)
    {
        p += _border;
        for(int j = _border; j < _width - _border; ++j)
        {
            if(*p < _verify_imin)
            {
                ++dark_center;
            }
            else if(*p >= _verify_imax)
            {
                ++light_center;
            }

            ++p;
        }
        p += _border;
    }

    MLTK_DEBUG("\ndark_full=%d\nlight_full=%d\nabs(dark_full-light_full)=%d\n(dark_full+light_full)/_verify_full_threshold=%d",
        dark_full, light_full, std::abs(dark_full-light_full), ((dark_full+light_full)/_verify_full_threshold));
    MLTK_DEBUG("\ndark_center=%d\nlight_center=%d\nabs(dark_center-light_center)=%d\n(dark_center+light_center)/_verify_center_threshold=%d\n",
        dark_full, light_full, std::abs(dark_center-light_center), (dark_center+light_center)/_verify_center_threshold);
    
    if(std::abs(dark_full-light_full) > ((dark_full+light_full)/_verify_full_threshold))
    {
        return false;
    }
    if(std::abs(dark_center-light_center) > ((dark_center+light_center)/_verify_center_threshold))
    {
        return false;
    }

    return true;
}



/*************************************************************************************************/
void DataPreprocessor::crop_image(const uint8_t* src, uint8_t* dst) const
{
    const uint16_t height_diff = FINGERPRINT_READER_IMAGE_HEIGHT - _height;
    const uint16_t width_diff = FINGERPRINT_READER_IMAGE_WIDTH - _width;
    const uint16_t hborder = width_diff / 2;
    const uint16_t vborder = height_diff / 2;

    const uint8_t* s = src + vborder*FINGERPRINT_READER_IMAGE_WIDTH;
    uint8_t* d = dst;
    for(int i = _height; i > 0; --i)
    {
        memcpy(d, s + hborder, _width);
        d += _width;
        s += FINGERPRINT_READER_IMAGE_WIDTH;
    }
}

/*************************************************************************************************/
void DataPreprocessor::balance_colorspace(uint8_t* sample) const
{
    uint8_t imin = 255;
    uint8_t imax = 0;
    uint8_t* p;

    p = sample + _border*_width;
    for(int i = _border; i < _height - _border; ++i)
    {
        p += _border;
        for(int j = _border; j < _width - _border; ++j)
        {
            if(*p < _balance_threshold_max)
            {
                imax = std::max(imax, *p);
            }
            if(*p > _balance_threshold_min)
            {
                imin = std::min(imin, *p);
            }

            ++p;
        }
        p += _border;
    }

    const float norm_scaler = 255.f / (imax - imin);

    p = sample;
    for(int i = _height*_width; i > 0; --i)
    {
        const float val_norm = norm_scaler * (*p - imin);
        *p++ = (uint8_t)std::min(255.f, std::max(0.f, val_norm));
    }
}


/*************************************************************************************************/
void DataPreprocessor::sharpen_image(uint8_t* sample) const
{
    const uint8_t pad_height = (_sharpen_filter_height - 1) / 2;
    const uint8_t pad_width = (_sharpen_filter_width - 1) / 2;

    uint8_t* out = sample;

    // Conv2D, stride=1, padding=SAME
    for(int out_y = 0; out_y < _height; ++out_y)
    {
        const int in_y_origin = out_y - pad_height;
        for(int out_x = 0; out_x < _width; ++out_x)
        {
            const int in_x_origin = out_x - pad_width;

            int32_t acc = 0;
            const int8_t *filter_ptr = _sharpen_filter;

            for(int filter_y = 0; filter_y < _sharpen_filter_height; ++filter_y)
            {
                const int in_y = in_y_origin + filter_y;
                if(in_y < 0 || in_y >= _height)
                {
                    filter_ptr += _sharpen_filter_width;
                    continue;
                }

                const int in_y_offset = in_y * _width;
                for(int filter_x = 0; filter_x < _sharpen_filter_width; ++filter_x)
                {
                    const int in_x = in_x_origin + filter_x;
                    if(in_x >= 0 && in_x < _width)
                    {
                        const int32_t filter_val = *filter_ptr;
                        acc += filter_val * sample[in_y_offset + in_x];
                    }

                    ++filter_ptr;
                }
            }

            const int32_t norm_val = acc / _sharpen_gain;
            *out++ = std::min((int32_t)255, std::max((int32_t)0, norm_val));
        }
    }
}


/*************************************************************************************************/
bool DataPreprocessor::preprocess_sample(const uint8_t* unprocessed_sample, uint8_t* processed_sample) const
{
    crop_image(unprocessed_sample, processed_sample);
    balance_colorspace(processed_sample);
    sharpen_image(processed_sample);

    const uint16_t header[2] = {_width, _height};
    jlink_stream::write_all("preprocessed", header, sizeof(header));
    jlink_stream::write_all("preprocessed", processed_sample, _width*_height);

    return true;
}



} // namespace mltk 
