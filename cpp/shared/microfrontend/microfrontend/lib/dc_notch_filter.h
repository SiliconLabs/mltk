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
#ifndef MICROFRONTEND_LIB_DC_NOTCH_FILTER_H_
#define MICROFRONTEND_LIB_DC_NOTCH_FILTER_H_

#include <stdint.h>
#include <stdlib.h>

#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

struct DcNotchFilterState {
    int enable_dc_notch_filter; // 1 if DC notch filter is enabled
    int16_t prev_in;  // last input from previous block
    int16_t prev_out;  // last output from previos block
    int16_t dc_notch_coef; // DC notch filter coefficient k in Q(16,15) format, H(z) = (1 - z^-1)/(1 - k*z^-1)
};

/**
 * DC notch filter: computes out[n] = in[n] - in[n-1] + k*out[n-1]
 * 
 * @param state Working state of filter
 * @param in input for current block
 * @param out ouput for current block
 * @param num_samp block size in samples
 */
DLL_EXPORT void DcNotchFilterProcessSamples(struct DcNotchFilterState *state, const int16_t *in, int16_t *out, int num_samp);

DLL_EXPORT void DcNotchFilterReset(struct DcNotchFilterState *state);


#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MICROFRONTEND_LIB_DC_NOTCH_FILTER_H_
