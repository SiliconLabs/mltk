#include <algorithm>
#include "microfrontend/lib/frontend_util.h"
#include "audio_feature_generator_wrapper.hpp"


namespace mltk 
{


static const auto int8_dtype = py::format_descriptor<int8_t>::format();
static const auto uint16_dtype = py::format_descriptor<uint16_t>::format();
static const auto float_dtype = py::format_descriptor<float>::format();


#define GET_SETTING(var, name, dtype) verify_setting(settings, name); var = settings[name].cast<dtype>()
#define GET_INT(var, name) GET_SETTING(var, name, int)
#define GET_FLOAT(var, name) GET_SETTING(var, name, float)


static void verify_setting(const py::dict& settings, const std::string& name);
typedef void* (*PopulateFunc)(AudioFeatureGeneratorWrapper *self, const struct FrontendOutput& frontend_output, void* output);
static void* populate_int8_slice(AudioFeatureGeneratorWrapper *self, const struct FrontendOutput& frontend_output, void* output);
static void* populate_uint16_slice(AudioFeatureGeneratorWrapper *self, const struct FrontendOutput& frontend_output, void* output);
static void* populate_float_slice(AudioFeatureGeneratorWrapper *self, const struct FrontendOutput& frontend_output, void* output);
static void dynamic_scale_int8_spectrogram(const uint16_t* src, int8_t* dst, int length, int dynamic_quantize_range);


/*************************************************************************************************/
AudioFeatureGeneratorWrapper::AudioFeatureGeneratorWrapper(const py::dict& settings)
{
  FrontendConfig config;

  FrontendFillConfigWithDefaults(&config);

  GET_INT(config.window.size_ms, "fe.window_size_ms");
  GET_INT(config.window.step_size_ms, "fe.window_step_ms");
  GET_INT(config.filterbank.num_channels, "fe.filterbank_n_channels");
  GET_FLOAT(config.filterbank.lower_band_limit, "fe.filterbank_lower_band_limit");
  GET_FLOAT(config.filterbank.upper_band_limit, "fe.filterbank_upper_band_limit");

  GET_INT(config.noise_reduction.enable_noise_reduction, "fe.noise_reduction_enable");
  GET_INT(config.noise_reduction.smoothing_bits, "fe.noise_reduction_smoothing_bits");
  GET_FLOAT(config.noise_reduction.even_smoothing, "fe.noise_reduction_even_smoothing");
  GET_FLOAT(config.noise_reduction.odd_smoothing, "fe.noise_reduction_odd_smoothing");
  GET_FLOAT(config.noise_reduction.min_signal_remaining, "fe.noise_reduction_min_signal_remaining");

  GET_INT(config.pcan_gain_control.enable_pcan, "fe.pcan_enable");
  GET_FLOAT(config.pcan_gain_control.strength, "fe.pcan_strength");
  GET_FLOAT(config.pcan_gain_control.offset , "fe.pcan_offset");
  GET_INT(config.pcan_gain_control.gain_bits , "fe.pcan_gain_bits");

  GET_INT(config.log_scale.enable_log, "fe.log_scale_enable");
  GET_INT(config.log_scale.scale_shift, "fe.log_scale_shift");

  GET_INT(config.activity_detection.enable_activation_detection, "fe.activity_detection_enable");
  GET_FLOAT(config.activity_detection.alpha_a, "fe.activity_detection_alpha_a");
  GET_FLOAT(config.activity_detection.alpha_b, "fe.activity_detection_alpha_b");
  GET_FLOAT(config.activity_detection.arm_threshold, "fe.activity_detection_arm_threshold");
  GET_FLOAT(config.activity_detection.trip_threshold, "fe.activity_detection_trip_threshold");
  GET_INT(config.dc_notch_filter.enable_dc_notch_filter, "fe.dc_notch_filter_enable");
  GET_FLOAT(config.dc_notch_filter.coefficient, "fe.dc_notch_filter_coefficient");

  int sample_rate_hz, window_size_ms, window_step_ms, sample_length_ms;
  GET_INT(sample_rate_hz, "fe.sample_rate_hz");
  GET_INT(_n_channels, "fe.filterbank_n_channels");
  GET_INT(window_size_ms, "fe.window_size_ms");
  GET_INT(window_step_ms, "fe.window_step_ms");
  GET_INT(sample_length_ms, "fe.sample_length_ms");

  float quantize_dynamic_scale_range_db;
  int quantize_dynamic_scale_enable;
  GET_INT(quantize_dynamic_scale_enable, "fe.quantize_dynamic_scale_enable");
  GET_FLOAT(quantize_dynamic_scale_range_db, "fe.quantize_dynamic_scale_range_db");
  if(quantize_dynamic_scale_enable)
  {
    // dynamic_range = quantize_dynamic_scale_range_db*(2^log_scale_shift)*ln(10)/20
    #define LN10_DIV_20 0.11512925465
    _dynamic_quantize_range = (int)(quantize_dynamic_scale_range_db * (float)(1 << config.log_scale.scale_shift) * LN10_DIV_20);
  } 
  else 
  {
    _dynamic_quantize_range = 0;
  }

  _sample_length = (sample_rate_hz * sample_length_ms) / 1000;
  _window_size = (window_size_ms * sample_rate_hz) / 1000;
  _window_step = (window_step_ms * sample_rate_hz) / 1000;
  _n_features = ((_sample_length - _window_size) / _window_step) + 1;

  if (!FrontendPopulateState(&config, &_frontend_state, sample_rate_hz)) 
  {
    throw std::invalid_argument("Failed to populate frontend state");
  }
}

/*************************************************************************************************/
AudioFeatureGeneratorWrapper::~AudioFeatureGeneratorWrapper()
{
  FrontendFreeStateContents(&_frontend_state);
}

/*************************************************************************************************/
void AudioFeatureGeneratorWrapper::process_sample(const py::array_t<int16_t>& input, py::array& output)
{
  const auto input_buf = input.request();
  auto output_buf = output.request();
  const auto shape = output_buf.shape;
  const auto dtype = output_buf.format;
  PopulateFunc populate_func;

  if(input_buf.ndim != 1)
  {
    throw std::invalid_argument("Input sample must be 1D array");
  }
  if(input_buf.size != _sample_length)
  {
    throw std::invalid_argument("Input sample must contain " + std::to_string(_sample_length) + " elements");
  }
  if(output_buf.ndim != 2)
  {
    throw std::invalid_argument("Output must be 2D array");
  }
  if(shape[0] != _n_features || shape[1] != _n_channels)
  {
    throw std::invalid_argument("Output must have shape" + std::to_string(_n_features) + "x" + std::to_string(_n_channels));
  }

  auto audio_ptr = static_cast<int16_t*>(input_buf.ptr);
  void* output_ptr = output_buf.ptr;
  void* quantize_tmp_buffer = nullptr;


  if(dtype == int8_dtype)
  {
    // If we're using dynamic quantization, 
    // then populated each slice as uint16 and at the end do the quantization
    if(_dynamic_quantize_range > 0)
    {
      populate_func = &populate_uint16_slice;
      output_ptr = quantize_tmp_buffer = malloc(sizeof(uint16_t) * _n_features * _n_channels);
    }
    else 
    {
      populate_func = &populate_int8_slice;
    }
    
  }
  else if(dtype == uint16_dtype)
  {
    populate_func = &populate_uint16_slice;
  }
  else if(dtype == float_dtype)
  {
    populate_func = &populate_float_slice;
  }
  else
  {
    throw std::invalid_argument("Output data type must be a int8, uint16, or float32");
  }


  FrontendReset(&_frontend_state);

  int samples_processed = 0;
  for(int i = _n_features; i > 0; --i)
  {
    size_t num_samples_read;
    const auto frontend_output = FrontendProcessSamples(
      &_frontend_state, 
      audio_ptr, 
      _sample_length - samples_processed, 
      &num_samples_read
    );
    samples_processed += num_samples_read;
    audio_ptr += num_samples_read;
    output_ptr = populate_func(this, frontend_output, output_ptr);
  }

  // If we're using dynamic quantization,
  // then convert the uint16 spectrogram in the tmp buffer
  // to int8 using dynamic quantization
  if(quantize_tmp_buffer != nullptr)
  {
    dynamic_scale_int8_spectrogram(
      (const uint16_t*)quantize_tmp_buffer, 
      (int8_t*)output_buf.ptr, 
      _n_features * _n_channels, 
      this->_dynamic_quantize_range
    );
    free(quantize_tmp_buffer);
  }

}

/*************************************************************************************************/
bool AudioFeatureGeneratorWrapper::activity_was_detected()
{
  return ActivityDetectionTripped(&_frontend_state.activity_detection);
}



/*************************************************************************************************/
static void verify_setting(const py::dict& settings, const std::string& name)
{
  if(!settings.contains(name))
  {
    throw std::invalid_argument("Expected AudioFeatureGenerator setting not found: " + name);
  }
}

/*************************************************************************************************
 * Refer to:
 * https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/micro_speech/micro_features/micro_features_generator.cc#L84
 * and sli_ml_audio_feature_generation_get_features_quantized()
 * for more details on  what is going on here
 */
static void* populate_int8_slice(AudioFeatureGeneratorWrapper *self, const struct FrontendOutput& frontend_output, void* output)
{
// Feature range min and max, used for determining valid range to quantize from
#define SL_ML_AUDIO_FEATURE_GENERATION_QUANTIZE_FEATURE_RANGE_MIN      0
#define SL_ML_AUDIO_FEATURE_GENERATION_QUANTIZE_FEATURE_RANGE_MAX      666    

  const uint16_t* src = frontend_output.values;
  int8_t *dst = static_cast<int8_t*>(output);
  for (size_t i = frontend_output.size; i > 0; --i) 
  {
    const int32_t value_scale = 256; 
     const uint16_t range_min = SL_ML_AUDIO_FEATURE_GENERATION_QUANTIZE_FEATURE_RANGE_MIN;
    const uint16_t range_max = SL_ML_AUDIO_FEATURE_GENERATION_QUANTIZE_FEATURE_RANGE_MAX;
    const int32_t value_div = range_max - range_min;
    int32_t value = (((*src++ - range_min) * value_scale) + (value_div / 2))
                    / value_div;
    value -= 128;
    if (value < -128) 
    {
      value = -128;
    }
    if (value > 127) 
    {
      value = 127;
    }
    *dst++ = (int8_t)value;
  }

  return dst;
}


/*************************************************************************************************/
static void* populate_uint16_slice(AudioFeatureGeneratorWrapper *self, const struct FrontendOutput& frontend_output, void* output)
{
  const uint16_t* src = frontend_output.values;
  uint16_t *dst = static_cast<uint16_t*>(output);
  for (size_t i = frontend_output.size; i > 0; --i) 
  {
    *dst++ = *src++;
  }
  return dst;
}

/*************************************************************************************************/
static void* populate_float_slice(AudioFeatureGeneratorWrapper *self, const struct FrontendOutput& frontend_output, void* output)
{
  const uint16_t* src = frontend_output.values;
  float *dst = static_cast<float*>(output);
  for (size_t i = frontend_output.size; i > 0; --i) 
  {
    const uint16_t value = *src++;
    *dst++ = static_cast<float>(value);
  }
  return dst;
}



/**************************************************************************************************/
static void dynamic_scale_int8_spectrogram(const uint16_t* src, int8_t* dst, int length, int dynamic_quantize_range)
{
  const uint16_t* ptr;

  // Find the maximum value in the uint16 spectrogram
  int32_t maxval = 0;

  ptr = src;
  for (size_t i = length; i > 0; --i) 
  {
    const int32_t value = (int32_t)*ptr++;
    maxval = std::max(value, maxval);
  }

  const int32_t minval = std::max(maxval - dynamic_quantize_range, 0);
  const int32_t val_range = std::max(maxval - minval, 1);

  // Scaling the uint16 spectrogram between -128 and +127 using the given range
  ptr = src;
  for (size_t i = length; i > 0; --i) 
  {
    int32_t value = (int32_t)*ptr++;
    value -= minval;
    value *= 255;
    value /= val_range;
    value -= 128;
    value = std::min(std::max(value, -128), 127);
    *dst++ = (int8_t)value;
  }
}


} // namespace mltk