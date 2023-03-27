/***************************************************************************//**
 * @file
 * @brief MVP Math Scale functions.
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
#ifndef SL_MATH_MVP_VECTOR_SCALE_H
#define SL_MATH_MVP_VECTOR_SCALE_H

#include "sl_status.h"
#include "sl_math_types.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/***************************************************************************//**
 * @addtogroup math_mvp_vector Vector functions
 * @{
 ******************************************************************************/

/**
 * @brief
 *   Scale a vector of 16-bits floats with a float16 scale.
 *
 * @details
 *   This function will perform the following operation: Z[i] = A[i] * scale.
 *   All vectors must be of equal length. If all vector buffers are 4-byte
 *   aligned, the function will operate twice as fast using MVP parallel
 *   processing.
 *   Maximum vector length is 1M (2^20), and 2M in the 4-byte aligned case.
 *
 * @param[in] input The input vector.
 * @param[in] scale The value by which to scale the vector.
 * @param[out] output Output vector, output Z.
 * @param[in] length Length of input and output vectors.
 *
 * @return
 *   @ref SL_STATUS_OK on success. On failure, an appropriate sl_status_t
 *   errorcode is returned.
 */
sl_status_t sl_math_mvp_vector_scale_f16(const float16_t *input, float16_t scale, float16_t *output, size_t length);

/** @} (end addtogroup math_mvp_vector) */

#ifdef __cplusplus
}
#endif

#endif // SL_MATH_MVP_VECTOR_SCALE_H
