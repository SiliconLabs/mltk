/***************************************************************************//**
 * @file
 * @brief Math type definitions.
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
#ifndef SL_MATH_TYPES_H
#define SL_MATH_TYPES_H


#if !defined(__arm__)
// This works around a build error in sl_common.h on Windows
#include "cmsis_compiler.h"
#define __STATIC_INLINE static inline
#endif

#include "sl_common.h"
#include "mltk_tflite_micro_helper.hpp"

#ifdef __arm__

#define GOTO_SLEEP() \
DWT->CTRL &= ~DWT_CTRL_CYCCNTENA_Msk; \
EMU_EnterEM1(); \
DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk

#define MLTK_PROFILER_INCREMENT_PARALLEL_PROG_COUNT(amount);
#define MLTK_PROFILER_INCREMENT_OPT_PROG_COUNT(amount);

#else // __arm__

#include <assert.h>

#include "sl_mvp_simulator.hpp"

#undef EFM_ASSERT
#define EFM_ASSERT assert

extern "C" void sli_mvp_increment_profiling_stat(const char* name, int32_t amount);
#define MLTK_PROFILER_INCREMENT_PARALLEL_PROG_COUNT(amount) sli_mvp_increment_profiling_stat("accelerator_parallel_loads", amount);
#define MLTK_PROFILER_INCREMENT_OPT_PROG_COUNT(amount) sli_mvp_increment_profiling_stat("accelerator_optimized_loads", amount);

#endif // __arm__



#ifdef __cplusplus
extern "C" {
#endif

/// @cond DO_NOT_INCLUDE_WITH_DOXYGEN
/***************************************************************************//**
 * @addtogroup math_types Data types
 * @{
 ******************************************************************************/

// Define the half precision floating point type, if not defined elsewhere.
// It will default to the 16 bits encoded in binary16 format.
#if __arm__
// float16_t is a half precision floating point type.
typedef __fp16 float16_t;
#endif // __arm__


/** @} (end addtogroup math_types) */
/// @endcond

#ifdef __cplusplus
}
#endif

#endif // SL_MATH_TYPES_H
