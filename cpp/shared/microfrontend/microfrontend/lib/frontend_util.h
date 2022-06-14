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
#ifndef MICROFRONTEND_LIB_FRONTEND_UTIL_H_
#define MICROFRONTEND_LIB_FRONTEND_UTIL_H_

#include "microfrontend/lib/utils.h"
#include "microfrontend/lib/filterbank_util.h"
#include "microfrontend/lib/frontend.h"
#include "microfrontend/lib/log_scale_util.h"
#include "microfrontend/lib/noise_reduction_util.h"
#include "microfrontend/lib/pcan_gain_control_util.h"
#include "microfrontend/lib/window_util.h"
#include "microfrontend/lib/activity_detection_util.h"
#include "microfrontend/lib/dc_notch_filter_util.h"


#ifdef __cplusplus
extern "C" {
#endif

struct FrontendConfig {
  struct WindowConfig window;
  struct FilterbankConfig filterbank;
  struct NoiseReductionConfig noise_reduction;
  struct PcanGainControlConfig pcan_gain_control;
  struct LogScaleConfig log_scale;
  struct ActivityDetectionConfig activity_detection;
  struct DcNotchFilterConfig dc_notch_filter;
};

// Fills the frontendConfig with "sane" defaults.
DLL_EXPORT void FrontendFillConfigWithDefaults(struct FrontendConfig* config);

// Allocates any buffers.
DLL_EXPORT int FrontendPopulateState(const struct FrontendConfig* config,
                          struct FrontendState* state, int sample_rate);

// Frees any allocated buffers.
DLL_EXPORT void FrontendFreeStateContents(struct FrontendState* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MICROFRONTEND_LIB_FRONTEND_UTIL_H_
