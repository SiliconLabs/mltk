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
#ifndef MICROFRONTEND_LIB_PCAN_GAIN_CONTROL_UTIL_H_
#define MICROFRONTEND_LIB_PCAN_GAIN_CONTROL_UTIL_H_

#include "microfrontend/lib/utils.h"
#include "microfrontend/lib/pcan_gain_control.h"

#define kWideDynamicFunctionBits 32
#define kWideDynamicFunctionLUTSize (4 * kWideDynamicFunctionBits - 3)

#ifdef __cplusplus
extern "C" {
#endif

struct PcanGainControlConfig {
  // set to false (0) to disable this module
  int enable_pcan;
  // gain normalization exponent (0.0 disables, 1.0 full strength)
  float strength;
  // positive value added in the normalization denominator
  float offset;
  // number of fractional bits in the gain
  int gain_bits;
};

DLL_EXPORT void PcanGainControlFillConfigWithDefaults(
    struct PcanGainControlConfig* config);

DLL_EXPORT int16_t PcanGainLookupFunction(const struct PcanGainControlConfig* config,
                               int32_t input_bits, uint32_t x);

DLL_EXPORT int PcanGainControlPopulateState(const struct PcanGainControlConfig* config,
                                 struct PcanGainControlState* state,
                                 uint32_t* noise_estimate,
                                 const int num_channels,
                                 const uint16_t smoothing_bits,
                                 const int32_t input_correction_bits);

DLL_EXPORT void PcanGainControlFreeStateContents(struct PcanGainControlState* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MICROFRONTEND_LIB_PCAN_GAIN_CONTROL_UTIL_H_
