
#include <string.h>
#include <assert.h>
#include "adpcm.h"

#ifndef MIN
#define MIN(x,y)  ((x) < (y) ? (x) : (y))
#endif /* ifndef MIN */
#ifndef MAX
#define MAX(x,y)  ((x) > (y) ? (x) : (y))
#endif /* ifndef MAX */

#define CLAMP(value, min, max) MAX(MIN(value, max), min)


static const int8_t STEP_INDICES[8] = 
{
  -1, -1, -1, -1, 2, 4, 6, 8,

};

static const int16_t STEPS[89] = 
{
  7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
  50, 55, 60, 66, 73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230,
  253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658, 724, 796, 876, 963,
  1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024,
  3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493,
  10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086,
  29794, 32767
};



void adpcm_decode_mono_data(const uint8_t* in, int16_t* out, int n_frames, adpcm_state_t* state)
{
  assert(n_frames % 2 == 0);

  for(int count = n_frames; count > 0; count -= 2)
  {
    adpcm_decode_sample(*in & 0x0F, state);
    *out++ = state->predicted_sample;
    adpcm_decode_sample(*in >> 4, state);
    *out++ = state->predicted_sample;
    ++in;
  }
}

void adpcm_encode_mono_data(const int16_t* in, uint8_t* out, int n_frames, adpcm_state_t* state)
{
  assert(n_frames % 2 == 0);

  for(int count = n_frames; count > 0; count -= 2)
  {
    const uint8_t nibble1 = adpcm_encode_sample(*in++, state);
    const uint8_t nibble2 = adpcm_encode_sample(*in++, state);
    *out++ = (nibble2 << 4 | nibble1);
  }
}


void adpcm_decode_sample(uint8_t sample, adpcm_state_t* state)
{
  int16_t step = STEPS[state->step_index];

  int diff = step >> 3;
  if(sample & 1)
      diff += step >> 2;
  if(sample & 2)
      diff += step >> 1;
  if(sample & 4)
      diff += step;
  if(sample & 8)
      diff = -diff;
  
  state->predicted_sample = CLAMP(state->predicted_sample + diff, -32768, 32767);
  state->step_index = CLAMP(state->step_index + STEP_INDICES[sample & 7], 0, 88);
}


uint8_t adpcm_encode_sample(int16_t sample, adpcm_state_t* state)
{
  int16_t step = STEPS[state->step_index];

  int sample_diff = sample - state->predicted_sample;
  uint8_t encoded_sample = (sample_diff < 0) ? 8 : 0;

  if(encoded_sample)
    sample_diff = -sample_diff;
  
  int diff = step >> 3;
  if(sample_diff >= step)
  {
      encoded_sample |= 4;
      sample_diff -= step;
      diff += step;
  }

  step >>= 1;
  if(sample_diff >= step)
  {
      encoded_sample |= 2;
      sample_diff -= step;
      diff += step;
  }

  step >>= 1;
  if(sample_diff >= step)
  {
      encoded_sample |= 1;
      diff += step;
  }

  if(encoded_sample & 8)
  {
      diff = -diff;
  }
    
  state->predicted_sample = CLAMP(state->predicted_sample + diff, -32768, 32767);
  state->step_index = CLAMP(state->step_index + STEP_INDICES[encoded_sample & 7], 0, 88);

  return encoded_sample;
}


