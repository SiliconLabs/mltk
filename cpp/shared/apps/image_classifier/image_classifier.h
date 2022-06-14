#pragma once

#include <stdint.h>



// These are parameters that are optionally embedded 
// into the .tflite model file
struct AppSettings
{
    float samplewise_norm_rescale = 0; // norm_img = img * rescale
    bool samplewise_norm_mean_and_std = false; // norm_img = (img - mean(img)) / std(img)
    bool verbose_inference_output = false; // enable verbose inference logging
    bool enable_inference = true; // Flag to enable inference
    uint32_t average_window_duration_ms = 1000; // Drop all inference results older than this value
    uint32_t minimum_count = 1; // The minimum number of inference results to average
    uint8_t detection_threshold = 160; // Minimum averaged model output threshold for a class to be considered detected, 0-255
    uint32_t suppression_count = 1; // The number of samples that are different than the last detected sample for a new detection to occur
    uint32_t latency_ms = 0; // This the amount of time in milliseconds between processing loop
    float activity_sensitivity = .5f;
};



extern AppSettings app_settings;
extern int category_count;

extern "C" void image_classifier_init(void);

const char *get_category_label(int index);




