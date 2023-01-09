#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


typedef struct 
{
  int8_t step_index;
  int16_t predicted_sample;
} adpcm_state_t;


void adpcm_decode_mono_data(const uint8_t* in, int16_t* out, int n_frames, adpcm_state_t* state);
void adpcm_encode_mono_data(const int16_t* sample, uint8_t* out, int n_frames, adpcm_state_t* state);

void adpcm_decode_sample(uint8_t in, adpcm_state_t* state);
uint8_t adpcm_encode_sample(int16_t sample, adpcm_state_t* state);

#ifdef __cplusplus
}
#endif