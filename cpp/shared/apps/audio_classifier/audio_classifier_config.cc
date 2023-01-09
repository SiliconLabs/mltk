
#include "tflite_micro_model/tflite_micro_model.hpp"
#include "audio_classifier_config.h"
#include "sl_ml_audio_feature_generation_config.h"
#include "cli_opts.hpp"



extern "C"
{

int SL_TFLITE_MODEL_AVERAGE_WINDOW_DURATION_MS;
int SL_TFLITE_MODEL_DETECTION_THRESHOLD;
int SL_TFLITE_MODEL_SUPPRESSION_MS;
int SL_TFLITE_MODEL_MINIMUM_COUNT;
float SL_TFLITE_MODEL_SENSITIVITY;
int SL_TFLITE_MODEL_VERBOSE_MODEL_OUTPUT_LOGS;
int SL_TFLITE_MODEL_INFERENCE_INTERVAL_MS;
mltk::StringList SL_TFLITE_MODEL_CLASSES;
mltk::Int32List SL_TFLITE_DETECTION_THRESHOLD_LIST;
int SL_TFLITE_MODEL_BAUD_RATE = 0;


bool mltk_app_settings_load_parameters(const void* tflite_flatbuffer)
{
    mltk::TfliteModelParameters model_parameters;

    if(!mltk::TfliteModelParameters::load_from_tflite_flatbuffer(tflite_flatbuffer, model_parameters))
    {
        return false;
    }

    if(!model_parameters.get("classes", SL_TFLITE_MODEL_CLASSES))
    {
        printf(".tflite does not define a \"classes\" parameter\n");
        return false;
    }


    SL_TFLITE_MODEL_AVERAGE_WINDOW_DURATION_MS = cli_opts.average_window_duration_ms;
    if(!cli_opts.average_window_duration_ms_provided)
    {
        model_parameters.get("average_window_duration_ms", SL_TFLITE_MODEL_AVERAGE_WINDOW_DURATION_MS);
    }

    SL_TFLITE_MODEL_DETECTION_THRESHOLD = cli_opts.detection_threshold;
    if(!cli_opts.detection_threshold_provided)
    {
        model_parameters.get("detection_threshold", SL_TFLITE_MODEL_DETECTION_THRESHOLD);
    }

    if(model_parameters.get("detection_threshold_list", SL_TFLITE_DETECTION_THRESHOLD_LIST))
    {
        if(SL_TFLITE_DETECTION_THRESHOLD_LIST.size() != SL_TFLITE_MODEL_CLASSES.size())
        {
            printf("The number of entries in the model parameter: 'detection_threshold_list' must match the number of classes\n");
            return false;
        }
    }


    SL_TFLITE_MODEL_SUPPRESSION_MS = cli_opts.suppression_ms;
    if(!cli_opts.suppression_ms_provided)
    {
        model_parameters.get("suppression_ms", SL_TFLITE_MODEL_SUPPRESSION_MS);
    }

    SL_TFLITE_MODEL_MINIMUM_COUNT = cli_opts.minimum_count;
    if(!cli_opts.minimum_count_provided)
    {
        model_parameters.get("minimum_count", SL_TFLITE_MODEL_MINIMUM_COUNT);
    }

    SL_TFLITE_MODEL_SENSITIVITY = cli_opts.sensitivity;
    if(!cli_opts.sensitivity_provided)
    {
        model_parameters.get("sensitivity", SL_TFLITE_MODEL_SENSITIVITY);
    }

    SL_TFLITE_MODEL_VERBOSE_MODEL_OUTPUT_LOGS = cli_opts.verbose;
    if(!cli_opts.verbose_provided)
    {
        model_parameters.get("verbose_model_output_logs", SL_TFLITE_MODEL_VERBOSE_MODEL_OUTPUT_LOGS);
    }

    SL_TFLITE_MODEL_INFERENCE_INTERVAL_MS = cli_opts.latency_ms;
    if(!cli_opts.latency_ms_provided)
    {
        model_parameters.get("latency_ms", SL_TFLITE_MODEL_INFERENCE_INTERVAL_MS);
    }

#ifndef __arm__
    // Ensure the loop interval is at least 10ms on Windows/Linux
    if(SL_TFLITE_MODEL_INFERENCE_INTERVAL_MS < 10)
    {
        SL_TFLITE_MODEL_INFERENCE_INTERVAL_MS = 10;
    }
#endif

    if(cli_opts.dump_audio)
    {
        SL_ML_AUDIO_FEATURE_GENERATION_DUMP_AUDIO = 1;
    }
    else
    {
        model_parameters.get("dump_audio", SL_ML_AUDIO_FEATURE_GENERATION_DUMP_AUDIO);
    }

    if(cli_opts.dump_raw_spectrograms)
    {
        SL_ML_AUDIO_FEATURE_GENERATION_DUMP_RAW_SPECTROGRAM = 1;
    }
    else
    {
        model_parameters.get("dump_raw_spectrograms", SL_ML_AUDIO_FEATURE_GENERATION_DUMP_RAW_SPECTROGRAM);
    }

    if(cli_opts.dump_spectrograms)
    {
        SL_ML_AUDIO_FEATURE_GENERATION_DUMP_QUANTIZED_SPECTROGRAM = 1;
    }
    else
    {
        model_parameters.get("dump_spectrograms", SL_ML_AUDIO_FEATURE_GENERATION_DUMP_QUANTIZED_SPECTROGRAM);
    }

    SL_ML_AUDIO_FEATURE_GENERATION_AUDIO_GAIN = cli_opts.volume_gain;
    if(!cli_opts.volume_gain_provided)
    {
        model_parameters.get("volume_gain", SL_ML_AUDIO_FEATURE_GENERATION_AUDIO_GAIN);
    }

#ifdef __arm__
    if(SL_ML_AUDIO_FEATURE_GENERATION_DUMP_AUDIO)
    {
        // Force the inference loop to 100ms when dumping audio
        SL_TFLITE_MODEL_INFERENCE_INTERVAL_MS = 100;
    }

    model_parameters.get("baud_rate", SL_TFLITE_MODEL_BAUD_RATE);
#endif

    printf("Application settings:\n");
    printf("average_window_duration_ms=%d\n", SL_TFLITE_MODEL_AVERAGE_WINDOW_DURATION_MS);
    printf("detection_threshold=%d\n", SL_TFLITE_MODEL_DETECTION_THRESHOLD);
    printf("suppression_ms=%d\n", SL_TFLITE_MODEL_SUPPRESSION_MS);
    printf("minimum_count=%d\n", SL_TFLITE_MODEL_MINIMUM_COUNT);
    printf("sensitivity=%4.2f\n", SL_TFLITE_MODEL_SENSITIVITY);
    printf("verbose_model_output_logs=%d\n", SL_TFLITE_MODEL_VERBOSE_MODEL_OUTPUT_LOGS);
    printf("latency_ms=%d\n", SL_TFLITE_MODEL_INFERENCE_INTERVAL_MS);
    printf("volume_gain=%d\n",  SL_ML_AUDIO_FEATURE_GENERATION_AUDIO_GAIN);
    printf("dump_audio=%d\n", SL_ML_AUDIO_FEATURE_GENERATION_DUMP_AUDIO);
    printf("dump_raw_spectrograms=%d\n",  SL_ML_AUDIO_FEATURE_GENERATION_DUMP_RAW_SPECTROGRAM);
    printf("dump_spectrograms=%d\n\n",  SL_ML_AUDIO_FEATURE_GENERATION_DUMP_QUANTIZED_SPECTROGRAM);

    return true;
}



} // extern "C"