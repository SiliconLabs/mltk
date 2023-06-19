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
#include "dc_notch_filter.h"
#include <stdio.h>



void DcNotchFilterProcessSamples(struct DcNotchFilterState *state, const int16_t *in, int16_t *out, int num_samp)
{
  // compute fist output samples
  const int16_t kDcNotchCoef = state->dc_notch_coef;
  int16_t d = in[0] - state->prev_in;
  int32_t p = kDcNotchCoef * state->prev_out;
  int16_t sum = d + (((p>>14)+1)>>1);
  out[0] = sum;
  // compute next num_samp-1 output samples
  for(int n=1; n<num_samp; n++)
  {
    d = in[n] - in[n-1];
    p = kDcNotchCoef*out[n-1];
    sum = d + (((p>>14)+1)>>1);
    out[n] = sum;
  }
  // update filter states
  state->prev_in = in[num_samp-1];
  state->prev_out = out[num_samp-1];
}

void DcNotchFilterReset(struct DcNotchFilterState *state) {
    state->prev_in = 0;
    state->prev_out = 0;
}