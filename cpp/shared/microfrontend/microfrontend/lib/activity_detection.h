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
#ifndef MICROFRONTEND_LIB_ACTIVITY_DETECTION_H_
#define MICROFRONTEND_LIB_ACTIVITY_DETECTION_H_

#include <stdint.h>
#include <stdlib.h>

#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ActivityDetectionState {
   int enable_activity_detection;
  int num_channels;
  int scale;
  int arm;
  int trip;
  int32_t filter_a; // filter state
  int32_t filter_b; // filter state
  int16_t alpha_a;  // filter coefficient Q(16,14)
  int16_t alpha_b;  // filter coefficient Q(16,14)
  int16_t arm_threshold;
  int16_t trip_threshold;
};

DLL_EXPORT void ActivityDetection(struct ActivityDetectionState *state, uint32_t* signal);

DLL_EXPORT void ActivityDetectionReset(struct ActivityDetectionState *state);

DLL_EXPORT int ActivityDetectionTripped(struct ActivityDetectionState *state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MICROFRONTEND_LIB_ACTIVITY_DETECTION_H_
