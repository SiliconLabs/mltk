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
#ifndef MICROFRONTEND_LIB_FILTERBANK_H_
#define MICROFRONTEND_LIB_FILTERBANK_H_

#include <stdint.h>
#include <stdlib.h>

#include "microfrontend/lib/utils.h"
#include "microfrontend/sl_ml_fft.h"


#define kFilterbankBits 12

#ifdef __cplusplus
extern "C" {
#endif

struct FilterbankState {
  int num_channels;
  int start_index;
  int end_index;
  int16_t* channel_frequency_starts;
  int16_t* channel_weight_starts;
  int16_t* channel_widths;
  int16_t* weights;
  int16_t* unweights;
  uint64_t* work;
};

// Converts the relevant complex values of an FFT output into energy (the
// square magnitude).
DLL_EXPORT void FilterbankConvertFftComplexToEnergy(struct FilterbankState* state,
                                         struct complex_int16_t* fft_output,
                                         int32_t* energy);

// Computes the mel-scale filterbank on the given energy array. Output is cached
// internally - to fetch it, you need to call FilterbankSqrt.
DLL_EXPORT void FilterbankAccumulateChannels(struct FilterbankState* state,
                                  const int32_t* energy);

// Applies an integer square root to the 64 bit intermediate values of the
// filterbank, and returns a pointer to them. Memory will be invalidated the
// next time FilterbankAccumulateChannels is called.
DLL_EXPORT uint32_t* FilterbankSqrt(struct FilterbankState* state, int scale_down_shift);

DLL_EXPORT void FilterbankReset(struct FilterbankState* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MICROFRONTEND_LIB_FILTERBANK_H_
