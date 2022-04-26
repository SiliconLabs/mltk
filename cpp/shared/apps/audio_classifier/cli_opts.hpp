#pragma once 

#include <cstdint>
#include "logging/logging.hpp"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#ifndef __arm__
#include <string>
#endif

#ifndef VERBOSE
#define VERBOSE false
#define VERBOSE_PROVIDED false
#else 
#define VERBOSE_PROVIDED true
#endif


#ifndef WINDOW_MS
#define WINDOW_MS 1000
#define WINDOW_MS_PROVIDED false
#else 
#define WINDOW_MS_PROVIDED true
#endif

#ifndef THRESHOLD
#define THRESHOLD 185
#define THRESHOLD_PROVIDED false 
#else 
#define THRESHOLD_PROVIDED true
#endif

#ifndef SUPPRESSION_MS
#define SUPPRESSION_MS 1500
#define SUPPRESSION_MS_PROVIDED false
#else 
#define SUPPRESSION_MS_PROVIDED true
#endif 

#ifndef COUNT
#define COUNT 3
#define COUNT_PROVIDED false
#else 
#define COUNT_PROVIDED true
#endif

#ifndef VOLUME_GAIN
#define VOLUME_GAIN 1
#define VOLUME_GAIN_PROVIDED false
#else 
#define VOLUME_GAIN_PROVIDED true
#endif

#ifndef LATENCY_MS
#define LATENCY_MS 100
#define LATENCY_MS_PROVIDED false
#else 
#define LATENCY_MS_PROVIDED true
#endif

#ifndef SENSITIVITY
#define SENSITIVITY .5f
#define SENSITIVITY_PROVIDED false
#else 
#define SENSITIVITY_PROVIDED true
#endif


struct CliOpts
{
    bool verbose = VERBOSE;
    int32_t average_window_duration_ms = WINDOW_MS;
    uint8_t detection_threshold = THRESHOLD;
    int32_t suppression_ms = SUPPRESSION_MS;
    int32_t minimum_count = COUNT;
    int32_t latency_ms = LATENCY_MS;
    int32_t volume_gain = VOLUME_GAIN;
    float sensitivity = SENSITIVITY;
    const uint8_t* model_flatbuffer = nullptr;
    bool verbose_provided = VERBOSE_PROVIDED;
    bool average_window_duration_ms_provided = WINDOW_MS_PROVIDED;
    bool detection_threshold_provided = THRESHOLD_PROVIDED;
    bool suppression_ms_provided = SUPPRESSION_MS_PROVIDED;
    bool minimum_count_provided = COUNT_PROVIDED;
    bool latency_ms_provided = LATENCY_MS_PROVIDED;
    bool volume_gain_provided = VOLUME_GAIN_PROVIDED;
    bool sensitivity_provided = SENSITIVITY_PROVIDED;
    bool model_flatbuffer_provided = false;
    bool dump_audio = false;
    bool dump_raw_spectrograms = false;
    bool dump_spectrograms = false;

#ifndef __arm__
    ~CliOpts();
#endif
};


extern CliOpts cli_opts;



#ifndef __arm__
void parse_cli_opts();
#endif
