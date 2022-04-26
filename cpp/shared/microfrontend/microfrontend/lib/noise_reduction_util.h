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
#ifndef MICROFRONTEND_LIB_NOISE_REDUCTION_UTIL_H_
#define MICROFRONTEND_LIB_NOISE_REDUCTION_UTIL_H_

#include "microfrontend/lib/utils.h"
#include "microfrontend/lib/noise_reduction.h"

#ifdef __cplusplus
extern "C" {
#endif

struct NoiseReductionConfig {
  // set to false (0) to disable noise reduction
  int enable_noise_reduction;
  // scale the signal up by 2^(smoothing_bits) before reduction
  int smoothing_bits;
  // smoothing coefficient for even-numbered channels
  float even_smoothing;
  // smoothing coefficient for odd-numbered channels
  float odd_smoothing;
  // fraction of signal to preserve (1.0 disables this module)
  float min_signal_remaining;
};

// Populates the NoiseReductionConfig with "sane" default values.
DLL_EXPORT void NoiseReductionFillConfigWithDefaults(struct NoiseReductionConfig* config);

// Allocates any buffers.
DLL_EXPORT int NoiseReductionPopulateState(const struct NoiseReductionConfig* config,
                                struct NoiseReductionState* state,
                                int num_channels);

// Frees any allocated buffers.
DLL_EXPORT void NoiseReductionFreeStateContents(struct NoiseReductionState* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MICROFRONTEND_LIB_NOISE_REDUCTION_UTIL_H_
