
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
typedef void* (*PopulateFunc)(const struct FrontendOutput& frontend_output, void* output);
static void* populate_int8_slice(const struct FrontendOutput& frontend_output, void* output);
static void* populate_uint16_slice(const struct FrontendOutput& frontend_output, void* output);
static void* populate_float_slice(const struct FrontendOutput& frontend_output, void* output);


/*************************************************************************************************/
AudioFeatureGeneratorWrapper::AudioFeatureGeneratorWrapper(const py::dict& settings)
{
  FrontendConfig config;

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

  
  int sample_rate_hz, window_size_ms, window_step_ms, sample_length_ms;
  GET_INT(sample_rate_hz, "fe.sample_rate_hz");
  GET_INT(_n_channels, "fe.filterbank_n_channels");
  GET_INT(window_size_ms, "fe.window_size_ms");
  GET_INT(window_step_ms, "fe.window_step_ms");
  GET_INT(sample_length_ms, "fe.sample_length_ms");

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


  if(dtype == int8_dtype)
  {
    populate_func = &populate_int8_slice;
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


  void* output_ptr = output_buf.ptr;
  auto audio_ptr = static_cast<int16_t*>(input_buf.ptr);
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
    output_ptr = populate_func(frontend_output, output_ptr);
  }
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
static void* populate_int8_slice(const struct FrontendOutput& frontend_output, void* output)
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
static void* populate_uint16_slice(const struct FrontendOutput& frontend_output, void* output)
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
static void* populate_float_slice(const struct FrontendOutput& frontend_output, void* output)
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


} // namespace mltk