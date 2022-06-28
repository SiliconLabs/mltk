
#include "tflite_micro_model/tflite_micro_model.hpp"
#include "sl_ml_audio_feature_generation_config.h"
#include "ble_audio_classifier_config.h"



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
int SL_TFLITE_MODEL_CLASS_COUNT;


bool ble_audio_classifier_config_load(const void* tflite_flatbuffer)
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
    SL_TFLITE_MODEL_CLASS_COUNT = SL_TFLITE_MODEL_CLASSES.size();


    SL_TFLITE_MODEL_AVERAGE_WINDOW_DURATION_MS = 200;
    model_parameters.get("average_window_duration_ms", SL_TFLITE_MODEL_AVERAGE_WINDOW_DURATION_MS);

    SL_TFLITE_MODEL_DETECTION_THRESHOLD = 185;
    model_parameters.get("detection_threshold", SL_TFLITE_MODEL_DETECTION_THRESHOLD);

    if(model_parameters.get("detection_threshold_list", SL_TFLITE_DETECTION_THRESHOLD_LIST))
    {
        if(SL_TFLITE_DETECTION_THRESHOLD_LIST.size() != SL_TFLITE_MODEL_CLASSES.size())
        {
            printf("The number of entries in the model parameter: 'detection_threshold_list' must match the number of classes\n");
            return false;
        }
    }

    SL_TFLITE_MODEL_SUPPRESSION_MS = 10;
    model_parameters.get("suppression_ms", SL_TFLITE_MODEL_SUPPRESSION_MS);

    SL_TFLITE_MODEL_MINIMUM_COUNT = 1;
    model_parameters.get("minimum_count", SL_TFLITE_MODEL_MINIMUM_COUNT);

    SL_TFLITE_MODEL_SENSITIVITY = .5f;
    model_parameters.get("sensitivity", SL_TFLITE_MODEL_SENSITIVITY);

    SL_TFLITE_MODEL_VERBOSE_MODEL_OUTPUT_LOGS = 0;
    model_parameters.get("verbose_model_output_logs", SL_TFLITE_MODEL_VERBOSE_MODEL_OUTPUT_LOGS);

    SL_TFLITE_MODEL_INFERENCE_INTERVAL_MS = 10;
    model_parameters.get("latency_ms", SL_TFLITE_MODEL_INFERENCE_INTERVAL_MS);

    SL_ML_AUDIO_FEATURE_GENERATION_AUDIO_GAIN = 1;
    model_parameters.get("volume_gain", SL_ML_AUDIO_FEATURE_GENERATION_AUDIO_GAIN);


    printf("Application settings:\n");
    printf("average_window_duration_ms=%d\n", SL_TFLITE_MODEL_AVERAGE_WINDOW_DURATION_MS);
    printf("detection_threshold=%d\n", SL_TFLITE_MODEL_DETECTION_THRESHOLD);
    printf("suppression_ms=%d\n", SL_TFLITE_MODEL_SUPPRESSION_MS);
    printf("minimum_count=%d\n", SL_TFLITE_MODEL_MINIMUM_COUNT);
    printf("sensitivity=%4.2f\n", SL_TFLITE_MODEL_SENSITIVITY);
    printf("verbose_model_output_logs=%d\n", SL_TFLITE_MODEL_VERBOSE_MODEL_OUTPUT_LOGS);
    printf("latency_ms=%d\n", SL_TFLITE_MODEL_INFERENCE_INTERVAL_MS);
    printf("volume_gain=%d\n",  SL_ML_AUDIO_FEATURE_GENERATION_AUDIO_GAIN);

    return true;
}



} // extern "C"