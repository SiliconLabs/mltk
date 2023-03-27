/***************************************************************************//**
 * @file
 * @brief MVP Math Vector Sub function.
 *******************************************************************************
 * # License
 * <b>Copyright 2023 Silicon Laboratories Inc. www.silabs.com</b>
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
#include "sl_mvp.h"
#include "sl_mvp_util.h"
#include "sl_math_mvp_vector_sub.h"
#include "sl_math_mvp_matrix_sub.h"

sl_status_t sl_math_mvp_vector_sub_f16(const float16_t *input_a,
                                       const float16_t *input_b,
                                       float16_t *output,
                                       size_t num_elements)
{
  uint32_t len, remainder;
  uint32_t m, n, i, parallel;
  sli_mvp_datatype_t data_type;

  if (!input_a || !input_b || !output || !num_elements) {
    return SL_STATUS_INVALID_PARAMETER;
  }

  // Check if MVP parallel processing is possible.
  len = num_elements;
  parallel = 1U;
  remainder = 0U;
  data_type = SLI_MVP_DATATYPE_BINARY16;
  if ((((intptr_t)input_a & 3U) == 0U)
      && (((intptr_t)input_b & 3U) == 0U)
      && (((intptr_t)output & 3U) == 0U)
      && (len >= 2U)) {
    parallel = 2U;
    if (len & 1U ) {
      remainder++;
    }
    data_type = SLI_MVP_DATATYPE_COMPLEX_BINARY16;
    len /= 2U;
  }

  // Factorize len into m * n.
  if (len <= SLI_MVP_MAX_ROW_LENGTH) {
    m = 1U;
    n = len;
  } else {
    i = len;
    while (sli_mvp_util_factorize_number(i, 1024U, &m, &n) != SL_STATUS_OK) {
      i--;
      remainder += parallel;
    }
  }

  sli_math_mvp_matrix_sub_f16(input_a, input_b, output, m, n, data_type);

  // When factorization above is incomplete we handle "tail" elements here.
  i = num_elements - remainder;
  while (remainder--) {
    output[i] = input_a[i] - input_b[i];
    i++;
  }

  sli_mvp_wait_for_completion();

  return SL_STATUS_OK;
}
