#include <cstdio>
#include "tflite_model_parameters/tflite_model_parameters.hpp"
#include "sl_ml_audio_feature_generation_config.h"



template<typename T>
bool get_param(const mltk::TfliteModelParameters& model_parameters, const char *key, T& val)
{
    if(!model_parameters.get(key, val))
    {
        printf("Model parameter missing: %s\n", key);
        val = 0;
        return true;
    }
    else 
    {
        return false;
    }
}


template<typename T>
bool get_param_default(const mltk::TfliteModelParameters& model_parameters, const char *key, T& val, T default_val)
{
    if(!model_parameters.get(key, val))
    {
       val = default_val;
    }
    return false;
}

#define GET_INT(key, val) missing_params |= get_param(model_parameters, key, val);  printf("%s=%d\n", key, val)
#define GET_INT_DEFAULT(key, val, default_val) get_param_default(model_parameters, key, val, default_val);  printf("%s=%d\n", key, val)
#define GET_FLOAT(key, val) missing_params |= get_param(model_parameters, key, val);  printf("%s=%5.3f\n", key, val)
#define GET_FLOAT_DEFAULT(key, val, default_val) get_param_default(model_parameters, key, val, default_val);  printf("%s=%5.3f\n", key, val)

extern "C"
{


int   SL_ML_AUDIO_FEATURE_GENERATION_AUDIO_BUFFER_SIZE;
int   SL_ML_AUDIO_FEATURE_GENERATION_AUDIO_GAIN;
int   SL_ML_AUDIO_FEATURE_GENERATION_DUMP_RAW_SPECTROGRAM;
int   SL_ML_AUDIO_FEATURE_GENERATION_DUMP_QUANTIZED_SPECTROGRAM;
int   SL_ML_AUDIO_FEATURE_GENERATION_DUMP_AUDIO;
int   SL_TFLITE_MODEL_FE_SAMPLE_RATE_HZ;
int   SL_TFLITE_MODEL_FE_SAMPLE_LENGTH_MS;
int   SL_TFLITE_MODEL_FE_WINDOW_SIZE_MS;
int   SL_TFLITE_MODEL_FE_WINDOW_STEP_MS;
int   SL_TFLITE_MODEL_FE_FFT_LENGTH;
int   SL_TFLITE_MODEL_FE_FILTERBANK_N_CHANNELS;
float SL_TFLITE_MODEL_FE_FILTERBANK_LOWER_BAND_LIMIT;
float SL_TFLITE_MODEL_FE_FILTERBANK_UPPER_BAND_LIMIT;
int   SL_TFLITE_MODEL_FE_NOISE_REDUCTION_ENABLE;
int   SL_TFLITE_MODEL_FE_NOISE_REDUCTION_SMOOTHING_BITS;
float SL_TFLITE_MODEL_FE_NOISE_REDUCTION_EVEN_SMOOTHING;
float SL_TFLITE_MODEL_FE_NOISE_REDUCTION_ODD_SMOOTHING;
float SL_TFLITE_MODEL_FE_NOISE_REDUCTION_MIN_SIGNAL_REMAINING;
int   SL_TFLITE_MODEL_FE_PCAN_ENABLE;
float SL_TFLITE_MODEL_FE_PCAN_STRENGTH;
float SL_TFLITE_MODEL_FE_PCAN_OFFSET;
int   SL_TFLITE_MODEL_FE_PCAN_GAIN_BITS;
int   SL_TFLITE_MODEL_FE_LOG_SCALE_ENABLE;
int   SL_TFLITE_MODEL_FE_LOG_SCALE_SHIFT;
int   SL_TFLITE_MODEL_FE_ACTIVITY_DETECTION_ENABLE;
float SL_TFLITE_MODEL_FE_ACTIVITY_DETECTION_ALPHA_A;
float SL_TFLITE_MODEL_FE_ACTIVITY_DETECTION_ALPHA_B;
float SL_TFLITE_MODEL_FE_ACTIVITY_DETECTION_ARM_THRESHOLD;
float SL_TFLITE_MODEL_FE_ACTIVITY_DETECTION_TRIP_THRESHOLD;
int   SL_TFLITE_MODEL_FE_DC_NOTCH_FILTER_ENABLE;
float SL_TFLITE_MODEL_FE_DC_NOTCH_FILTER_COEFFICIENT;
int   SL_TFLITE_MODEL_FE_QUANTIZE_DYNAMIC_SCALE_ENABLE;
float SL_TFLITE_MODEL_FE_QUANTIZE_DYNAMIC_SCALE_RANGE_DB;
float SL_TFLITE_MODEL_SAMPLEWISE_NORM_RESCALE;
int   SL_TFLITE_MODEL_SAMPLEWISE_NORM_MEAN_AND_STD;


bool mltk_sl_ml_audio_feature_generation_load_parameters(const void* tflite_flatbuffer)
{
    mltk::TfliteModelParameters model_parameters;
    bool missing_params = false;

    if(!mltk::TfliteModelParameters::load_from_tflite_flatbuffer(tflite_flatbuffer, model_parameters))
    {
        return false;
    }

    printf("\nAudioFeatureGenerator settings:\n");
    GET_INT("fe.sample_rate_hz", SL_TFLITE_MODEL_FE_SAMPLE_RATE_HZ);
    GET_INT("fe.sample_length_ms", SL_TFLITE_MODEL_FE_SAMPLE_LENGTH_MS);
    GET_INT("fe.window_size_ms", SL_TFLITE_MODEL_FE_WINDOW_SIZE_MS);
    GET_INT("fe.window_step_ms", SL_TFLITE_MODEL_FE_WINDOW_STEP_MS);
    GET_INT("fe.fft_length", SL_TFLITE_MODEL_FE_FFT_LENGTH);
    GET_INT("fe.filterbank_n_channels", SL_TFLITE_MODEL_FE_FILTERBANK_N_CHANNELS);
    GET_FLOAT("fe.filterbank_lower_band_limit", SL_TFLITE_MODEL_FE_FILTERBANK_LOWER_BAND_LIMIT);
    GET_FLOAT("fe.filterbank_upper_band_limit", SL_TFLITE_MODEL_FE_FILTERBANK_UPPER_BAND_LIMIT);
    GET_INT("fe.noise_reduction_enable", SL_TFLITE_MODEL_FE_NOISE_REDUCTION_ENABLE);
    GET_INT("fe.noise_reduction_smoothing_bits", SL_TFLITE_MODEL_FE_NOISE_REDUCTION_SMOOTHING_BITS);
    GET_FLOAT("fe.noise_reduction_even_smoothing", SL_TFLITE_MODEL_FE_NOISE_REDUCTION_EVEN_SMOOTHING);
    GET_FLOAT("fe.noise_reduction_odd_smoothing", SL_TFLITE_MODEL_FE_NOISE_REDUCTION_ODD_SMOOTHING);
    GET_FLOAT("fe.noise_reduction_min_signal_remaining", SL_TFLITE_MODEL_FE_NOISE_REDUCTION_MIN_SIGNAL_REMAINING);
    GET_INT("fe.pcan_enable", SL_TFLITE_MODEL_FE_PCAN_ENABLE);
    GET_FLOAT("fe.pcan_strength", SL_TFLITE_MODEL_FE_PCAN_STRENGTH);
    GET_FLOAT("fe.pcan_offset", SL_TFLITE_MODEL_FE_PCAN_OFFSET);
    GET_INT("fe.pcan_gain_bits", SL_TFLITE_MODEL_FE_PCAN_GAIN_BITS);
    GET_INT("fe.log_scale_enable", SL_TFLITE_MODEL_FE_LOG_SCALE_ENABLE);
    GET_INT("fe.log_scale_shift", SL_TFLITE_MODEL_FE_LOG_SCALE_SHIFT);
    GET_INT_DEFAULT("fe.activity_detection_enable", SL_TFLITE_MODEL_FE_ACTIVITY_DETECTION_ENABLE, 0);
    GET_FLOAT_DEFAULT("fe.activity_detection_alpha_a", SL_TFLITE_MODEL_FE_ACTIVITY_DETECTION_ALPHA_A, 0.f);
    GET_FLOAT_DEFAULT("fe.activity_detection_alpha_b", SL_TFLITE_MODEL_FE_ACTIVITY_DETECTION_ALPHA_B, 0.f);
    GET_FLOAT_DEFAULT("fe.activity_detection_arm_threshold", SL_TFLITE_MODEL_FE_ACTIVITY_DETECTION_ARM_THRESHOLD, 0.f);
    GET_FLOAT_DEFAULT("fe.activity_detection_trip_threshold", SL_TFLITE_MODEL_FE_ACTIVITY_DETECTION_TRIP_THRESHOLD, 0.f);
    GET_INT_DEFAULT("fe.dc_notch_filter_enable", SL_TFLITE_MODEL_FE_DC_NOTCH_FILTER_ENABLE, 0);
    GET_FLOAT_DEFAULT("fe.dc_notch_filter_coefficient", SL_TFLITE_MODEL_FE_DC_NOTCH_FILTER_COEFFICIENT, 0.f);

    GET_INT_DEFAULT("fe.quantize_dynamic_scale_enable", SL_TFLITE_MODEL_FE_QUANTIZE_DYNAMIC_SCALE_ENABLE, 0);
    GET_FLOAT_DEFAULT("fe.quantize_dynamic_scale_range_db", SL_TFLITE_MODEL_FE_QUANTIZE_DYNAMIC_SCALE_RANGE_DB, 0.f);

    SL_ML_AUDIO_FEATURE_GENERATION_AUDIO_GAIN = 1;

    // Here, we use a 1s audio buffer.
    // For a real application this is overkill,
    // but this is required when dumping the audio and spectrograms
    SL_ML_AUDIO_FEATURE_GENERATION_AUDIO_BUFFER_SIZE = SL_TFLITE_MODEL_FE_SAMPLE_RATE_HZ*1;
    printf("fe.audio_buffer_size = %d\n", SL_ML_AUDIO_FEATURE_GENERATION_AUDIO_BUFFER_SIZE);
    
    GET_FLOAT_DEFAULT("samplewise_norm.rescale", SL_TFLITE_MODEL_SAMPLEWISE_NORM_RESCALE, 0.f);
    GET_INT_DEFAULT("samplewise_norm.mean_and_std", SL_TFLITE_MODEL_SAMPLEWISE_NORM_MEAN_AND_STD, 0);

    printf("\n");

    return !missing_params;
}


} // extern "C"