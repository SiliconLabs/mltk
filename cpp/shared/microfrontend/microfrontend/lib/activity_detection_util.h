/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file has been modified by Silicon Labs.
==============================================================================*/
#ifndef ACTIVITY_DETECTION_UTIL_H_
#define ACTIVITY_DETECTION_UTIL_H_

#include "microfrontend/lib/utils.h"
#include "microfrontend/lib/activity_detection.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ActivityDetectionConfig {
    int enable_activation_detection; // set to false (0) to disable activity detection
    float alpha_a;        // filter a coefficient, Q(16,14)
    float alpha_b;        // filter b coefficient, Q(16,14)
    float arm_threshold;  // detector arm threshold, Q(16,12)
    float trip_threshold; // detector trip threshols, Q(16,12)
};

DLL_EXPORT void ActivityDetectionFillConfigWithDefaults(struct ActivityDetectionConfig* config);

DLL_EXPORT void ActivityDetectionConfig(const struct ActivityDetectionConfig* config, int num_channels, struct ActivityDetectionState* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ACTIVITY_DETECTION_UTIL_H_
