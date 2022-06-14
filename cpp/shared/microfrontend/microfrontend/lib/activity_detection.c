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
#include "activity_detection.h"
#include <stdio.h>

// 1-real pole IIR filter: computes out = (1-k)*in + k*out 
static void OnePoleIIR(
  int32_t in,   
  int32_t *out, 
  int16_t k     // Q(16,8)
)
{
  int32_t state = *out;
  int16_t one_minus_k = (1<<8) - k;
  int64_t p1 = one_minus_k*in;
  int64_t p2 = k*state;
  int32_t sum = (((p1+p2)>>7)+1)>>1;
  *out = sum;
}

// spike detector
static void SpikeDetect(struct ActivityDetectionState *state)
{
  if(state->arm==0)
  {
    // (filter_a/filter_b) < arm_threshold ?
    int32_t threshold_a = ((int64_t)state->filter_b*state->arm_threshold)>>12; 
    if(state->filter_a < threshold_a)
    {
      // arm detector
      state->arm = 1;
    }
  }
  else
  {
    // (fiter_a/filter_b) > trip_threshold ?
    int32_t threshold_b = ((int64_t)state->filter_b*state->trip_threshold)>>12; 
    if(state->filter_a > threshold_b)
    {
      // trip detector
      state->arm = 0;
      state->trip = 1; // trip is set until ActivityDetectionTripped() is called
    }
  }
}

//static int run_once = 0;
//static FILE *fp;

void ActivityDetection(struct ActivityDetectionState *state, uint32_t* signal)
{  
  //if(!run_once)
  //{
  //  fp = fopen("C:\\tmp\\debug.txt","w");  
  //  run_once = 1;
  //}

  // compute total power
  uint32_t x = 0;
  for (int i = 0; i < state->num_channels; ++i)
  {
    x += signal[i];
    //fprintf(fp, "%d,", signal[i]);
  }
  x>>=state->scale;

  // update filters
  OnePoleIIR(x, &state->filter_a, state->alpha_a);
  OnePoleIIR(x, &state->filter_b, state->alpha_b);
  //fprintf(fp, "%d,%d,%d\n", x, state->filter_a, state->filter_b );

  // Detection logic
  SpikeDetect(state);
}

void ActivityDetectionReset(struct ActivityDetectionState *state)
{
  state->filter_a = 0;
  state->filter_b = 0;
  state->arm = 0;
  state->trip = 0;
}

int ActivityDetectionTripped(struct ActivityDetectionState *state)
{
  const int retval = state->trip == 1;
  state->trip = 0;
  return retval;
}