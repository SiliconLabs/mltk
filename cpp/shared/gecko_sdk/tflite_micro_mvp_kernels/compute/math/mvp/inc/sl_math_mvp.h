/***************************************************************************//**
 * @file
 * @brief MVP Math functions.
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
#ifndef SL_MATH_MVP_H
#define SL_MATH_MVP_H

// Matrix
#include "sl_math_mvp_matrix_add.h"
#include "sl_math_mvp_matrix_mult.h"
#include "sl_math_mvp_matrix_scale.h"
#include "sl_math_mvp_matrix_sub.h"
#include "sl_math_mvp_matrix_transpose.h"
#include "sl_math_mvp_matrix_vector_mult.h"

// Vector
#include "sl_math_mvp_vector_abs.h"
#include "sl_math_mvp_vector_add.h"
#include "sl_math_mvp_vector_clamp.h"
#include "sl_math_mvp_vector_clip.h"
#include "sl_math_mvp_vector_copy.h"
#include "sl_math_mvp_vector_dot_product.h"
#include "sl_math_mvp_vector_fill.h"
#include "sl_math_mvp_vector_mult.h"
#include "sl_math_mvp_vector_negate.h"
#include "sl_math_mvp_vector_scale.h"
#include "sl_math_mvp_vector_sub.h"

/* *INDENT-OFF* */
/************ THIS SECTION IS FOR DOCUMENTATION ONLY !**********************//**
 * @addtogroup math_mvp MVP Math Library Introduction
 * @{

# Introduction
This user manual describes the math library that has been designed to utilize the
MVP hardware accelerator to speed up vector and matrix operations compared to
using the CPU for the same operations.

Similar or identical functions can be found in the
[ARM CMSIS-DSP](https://www.keil.com/pack/doc/CMSIS/DSP/html/index.html)
library.

The library contains only functions that can be accelerated on the MVP. The
MVP is limited to support 8-bit integers and 16-bit floats, and cannot
operate on other data types.

The library is divided into a number of functions covering a specific category:
@li Matrix functions
@li Vector functions

# Using the library
The library is released in source form.

The library functions are decleared separate header files, but for convenience,
the **sl_math_mvp.h** file contains everything the application needs to use the
library. To use the library, the application shoul include this file only.

 * @} end addtogroup math_mvp *************************************************/

#endif // SL_MATH_MVP_H
