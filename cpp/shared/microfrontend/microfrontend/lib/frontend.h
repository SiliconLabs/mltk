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
#ifndef MICROFRONTEND_LIB_FRONTEND_H_
#define MICROFRONTEND_LIB_FRONTEND_H_

#include <stdint.h>
#include <stdlib.h>


#include "microfrontend/lib/utils.h"
#include "microfrontend/sl_ml_fft.h"
#include "microfrontend/lib/filterbank.h"
#include "microfrontend/lib/log_scale.h"
#include "microfrontend/lib/noise_reduction.h"
#include "microfrontend/lib/pcan_gain_control.h"
#include "microfrontend/lib/window.h"
#include "microfrontend/lib/activity_detection.h"
#include "microfrontend/lib/dc_notch_filter.h"


#ifdef __cplusplus
extern "C" {
#endif

struct FrontendState {
  struct WindowState window;
  struct sli_ml_fft_state fft;
  struct FilterbankState filterbank;
  struct NoiseReductionState noise_reduction;
  struct PcanGainControlState pcan_gain_control;
  struct LogScaleState log_scale;
  struct ActivityDetectionState activity_detection;
  struct DcNotchFilterState dc_notch_filter;
};

struct FrontendOutput {
  const uint16_t* values;
  size_t size;
};

// Main entry point to processing frontend samples. Updates num_samples_read to
// contain the number of samples that have been consumed from the input array.
// Returns a struct containing the generated output. If not enough samples were
// added to generate a feature vector, the returned size will be 0 and the
// values pointer will be NULL. Note that the output pointer will be invalidated
// as soon as FrontendProcessSamples is called again, so copy the contents
// elsewhere if you need to use them later.
DLL_EXPORT struct FrontendOutput FrontendProcessSamples(struct FrontendState* state,
                                             const int16_t* samples,
                                             size_t num_samples,
                                             size_t* num_samples_read);

DLL_EXPORT void FrontendReset(struct FrontendState* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MICROFRONTEND_LIB_FRONTEND_H_
