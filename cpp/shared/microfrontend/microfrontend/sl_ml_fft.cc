/***************************************************************************//**
 * @file
 * @brief SL_ML_FFT 
 *******************************************************************************
 * # License
 * <b>Copyright 2021 Silicon Laboratories Inc. www.silabs.com</b>
 *******************************************************************************
 *
 * SPDX-License-Identifier: Zlib
 *
 * The licensor of this software is Silicon Laboratories Inc.
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 ******************************************************************************/
#include <cassert>
#include <cstdlib>

#include "microfrontend/sl_ml_fft.h"
//#include "microfrontend/config/sl_ml_frontend_config.h"
#include "CMSIS/DSP/Include/arm_const_structs.h"
#include "CMSIS/DSP/Include/arm_common_tables.h"
#include "CMSIS/DSP/Include/dsp/transform_functions.h"

// Helper Functions
// #define SL_CONCAT(A, B) A ## B
// #define CFFT_FUNCTION(N) SL_CONCAT(&arm_cfft_sR_q15_len, N)

// Define length of complex FFT
// #if SL_ML_FRONTEND_CONFIG_RFFT_LEN==8192
//   #define SL_ML_CFFT_LEN 4096
// #elif SL_ML_FRONTEND_CONFIG_RFFT_LEN==4096
//   #define SL_ML_CFFT_LEN 2048
// #elif SL_ML_FRONTEND_CONFIG_RFFT_LEN==2048
//   #define SL_ML_CFFT_LEN 1024
// #elif SL_ML_FRONTEND_CONFIG_RFFT_LEN==1024
//   #define SL_ML_CFFT_LEN 512
// #elif SL_ML_FRONTEND_CONFIG_RFFT_LEN==512
//   #define SL_ML_CFFT_LEN 256
// #elif SL_ML_FRONTEND_CONFIG_RFFT_LEN==256
//   #define SL_ML_CFFT_LEN 128
// #elif SL_ML_FRONTEND_CONFIG_RFFT_LEN==128
//   #define SL_ML_CFFT_LEN 64
// #elif SL_ML_FRONTEND_CONFIG_RFFT_LEN==64
//   #define SL_ML_CFFT_LEN 32
// #elif SL_ML_FRONTEND_CONFIG_RFFT_LEN==32
//   #define SL_ML_CFFT_LEN 16
// #endif 


// FFT Input and Output buffers
// static int16_t *input_buffer[SL_ML_FRONTEND_CONFIG_RFFT_LEN];
// static int16_t *output_buffer[SL_ML_FRONTEND_CONFIG_RFFT_LEN * 2];

// RFFT instance
static arm_rfft_instance_q15 rfft_instance;

// Real Coefficient Tables 
static q15_t* __ALIGNED(4) real_coeff_table_A = nullptr;
static q15_t* __ALIGNED(4) real_coeff_table_B = nullptr;

/*******************************************************************************
 *  The real coefficient tables are used by the CMSIS rfft implementation during 
 *  the split operation. The number of coefficients needed for this operation is
 *  dependent on the length of the FFT. 
 * 
 *  The CMSIS DSP library provides the coefficient tables as large tables 
 *  supporting different size FFT's up to 8192. In order to minimize ROM usage 
 *  for shorter FFT lengths, this function re-generates the tables specific 
 *  to the length of the FFT.  
 ******************************************************************************/
static void generate_real_coeff_tables(int n){

  // Generate the real coefficient tables 
  for (int i = 0; i < n; i++){
     real_coeff_table_A[2 * i]     = (q15_t)roundf((0.5 * ( 1.0 - sinf(2 * PI / (float) (2 * n) * (float) i))) * powf(2,15));
     real_coeff_table_A[2 * i + 1] = (q15_t)roundf((0.5 * (-1.0 * cosf(2 * PI / (float) (2 * n) * (float) i))) * powf(2,15));
     
     real_coeff_table_B[2 * i]     = (q15_t)roundf((0.5 * (1.0 + sinf(2 * PI / (float) (2 * n) * (float) i))) * powf(2,15));
     real_coeff_table_B[2 * i + 1] = (q15_t)roundf((0.5 * (1.0 * cosf(2 * PI / (float) (2 * n) * (float) i))) * powf(2,15));;
  }
}

/**************************************************************************//**
 * Compute real FFT 
 *****************************************************************************/
void sli_ml_fft_compute(struct sli_ml_fft_state* state, const int16_t* input,
                int input_scale_shift) {
  const size_t input_size = state->input_size;
  const size_t fft_size = state->fft_size;

  int16_t* fft_input = state->input;
  
  // Scale input by the given shift.
  size_t i;
  for (i = 0; i < input_size; ++i) {
    fft_input[i] = static_cast<int16_t>(static_cast<uint16_t>(input[i])
                                        << input_scale_shift);
  }
  
  // Zero out whatever else remains in the top part of the input.
  for (; i < fft_size; ++i) {
    fft_input[i] = 0;
  }

  // Apply the FFT.
  arm_rfft_q15(&rfft_instance, (q15_t*)fft_input, (q15_t*)state->output);
}

/**************************************************************************//**
 * Initialize FFT state
 *****************************************************************************/
sl_status_t sli_ml_fft_init(struct sli_ml_fft_state* state, size_t input_size) {

  if (input_size < 16 || input_size > 8192){
    return SL_STATUS_INVALID_CONFIGURATION;
  }

  int rfft_size = 32;
  for(; rfft_size <= 8192; rfft_size <<=1)
  {
    if(input_size <= rfft_size)
    {
      break;
    }
  }
  
  // Initialize FFT struct
  state->input_size = input_size;
  state->fft_size = rfft_size;

  state->input = static_cast<int16_t*>(malloc(rfft_size * sizeof(int16_t)));
  if(state->input == nullptr)
  {
    return SL_STATUS_ALLOCATION_FAILED;
  }
  state->output = static_cast<complex_int16_t*>(malloc(rfft_size * sizeof(complex_int16_t)));
  if(state->output == nullptr)
  {
    return SL_STATUS_ALLOCATION_FAILED;
  }

  real_coeff_table_A  = static_cast<int16_t*>(malloc(rfft_size * sizeof(int16_t)));
  if(real_coeff_table_A == nullptr)
  {
    return SL_STATUS_ALLOCATION_FAILED;
  }
  real_coeff_table_B  = static_cast<int16_t*>(malloc(rfft_size * sizeof(int16_t)));
  if(real_coeff_table_B == nullptr)
  {
    return SL_STATUS_ALLOCATION_FAILED;
  }

  // Generate tables
  generate_real_coeff_tables(rfft_size/2);

  // Set up RFFT instance
  rfft_instance.fftLenReal = (uint16_t) rfft_size;
  rfft_instance.pTwiddleAReal = (q15_t *) real_coeff_table_A;
  rfft_instance.pTwiddleBReal = (q15_t *) real_coeff_table_B;
  rfft_instance.ifftFlagR = 0U;
  rfft_instance.bitReverseFlagR = 1U;

  // Set modifier to 1 when using the autogenerate coeftables
  rfft_instance.twidCoefRModifier = 1U;
  const int cfft_len = rfft_size / 2;
  switch(cfft_len)
  {
  case 16:
    rfft_instance.pCfft = &arm_cfft_sR_q15_len16;
    break;
  case 32:
    rfft_instance.pCfft = &arm_cfft_sR_q15_len32;
    break;
  case 64:
    rfft_instance.pCfft = &arm_cfft_sR_q15_len64;
    break;
  case 128:
    rfft_instance.pCfft = &arm_cfft_sR_q15_len128;
    break;
  case 256:
    rfft_instance.pCfft = &arm_cfft_sR_q15_len256;
    break;
  case 512:
    rfft_instance.pCfft = &arm_cfft_sR_q15_len512;
    break;
  case 1024:
    rfft_instance.pCfft = &arm_cfft_sR_q15_len1024;
    break;
  case 2048:
    rfft_instance.pCfft = &arm_cfft_sR_q15_len2048;
    break;
  case 4096:
    rfft_instance.pCfft = &arm_cfft_sR_q15_len4096;
    break;
  default:
    assert(!"Bad cfft length");
  }

  return SL_STATUS_OK;
}

void sli_ml_fft_deinit(struct sli_ml_fft_state* state)
{
  if(state->input != nullptr)
  {
    free(state->input);
    state->input = nullptr;
  }
  if(state->output != nullptr)
  {
    free(state->output);
    state->output = nullptr;
  }
  if(real_coeff_table_A != nullptr)
  {
    free(real_coeff_table_A);
    real_coeff_table_A = nullptr;
  }
  if(real_coeff_table_B != nullptr)
  {
    free(real_coeff_table_B);
    real_coeff_table_B = nullptr;
  }
}

/**************************************************************************//**
 * Clear FFT buffers
 *****************************************************************************/
void sli_ml_fft_reset(struct sli_ml_fft_state* state){
  memset(state->input, 0, state->fft_size * sizeof(*state->input));
  memset(state->output, 0, (state->fft_size / 2 + 1) * sizeof(*state->output));
}



