/***************************************************************************//**
 * @file
 * @brief MVP Math Mul functions.
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
#include "sl_math_mvp.h"
#include "sl_mvp_util.h"
#include "sl_mvp_power.h"
#include "sl_mvp_program_area.h"
#include "sl_common.h"

#ifndef USE_MVP_PROGRAMBUILDER
#define USE_MVP_PROGRAMBUILDER    0
#endif

sl_status_t sl_math_mvp_matrix_mult_f16(const sl_math_matrix_f16_t *input_a, const sl_math_matrix_f16_t *input_b, sl_math_matrix_f16_t *output)
{
  if (!input_a || !input_b || !output) {
    return SL_STATUS_INVALID_PARAMETER;
  }
  if (input_a->num_cols != input_b->num_rows) {
    return SL_STATUS_INVALID_PARAMETER;
  }
  if (output->num_cols != input_b->num_cols || output->num_rows != input_a->num_rows) {
    return SL_STATUS_INVALID_PARAMETER;
  }
  uint32_t a_rows = input_a->num_rows;
  uint32_t a_cols = input_a->num_cols;
  uint32_t b_rows = input_b->num_rows;
  uint32_t b_cols = input_b->num_cols;
  uint32_t z_rows = output->num_rows;
  uint32_t z_cols = output->num_cols;
  sli_mvp_datatype_t data_type = SLI_MVP_DATATYPE_BINARY16;
  bool parallel = false;

  // Check if all pointers are 4-byte aligned
  if (sli_mvp_util_is_pointer_word_aligned((void *)input_a->data)
      && sli_mvp_util_is_pointer_word_aligned((void *)input_b->data)
      && sli_mvp_util_is_pointer_word_aligned((void *)output->data)) {
    // Check that the relevant dimensions are divisible by 2, only need to check input_b because we know output has the same number of columns.
    if ((b_cols % 2 == 0)) {
      // Enable parallel processing
      parallel = true;
      b_cols /= 2;
      z_cols /= 2;
      data_type = SLI_MVP_DATATYPE_COMPLEX_BINARY16;
    }
  }
  const int matrix_x = SLI_MVP_ARRAY(0);
  const int matrix_y = SLI_MVP_ARRAY(1);
  const int matrix_z = SLI_MVP_ARRAY(2);

  #if USE_MVP_PROGRAMBUILDER

  sl_status_t status;
  sli_mvp_program_context_t *p = sli_mvp_get_program_area_context();
  sli_mvp_pb_init_program(p);
  sli_mvp_pb_begin_program(p);

  // Configure the matrices
  sli_mvp_pb_config_matrix(p->p, matrix_x, (void*)input_a->data, SLI_MVP_DATATYPE_BINARY16, a_rows, a_cols, &status);
  sli_mvp_pb_config_matrix(p->p, matrix_y, (void*)input_b->data, data_type, b_rows, b_cols, &status);
  sli_mvp_pb_config_matrix(p->p, matrix_z, (void*)output->data, data_type, z_rows, z_cols, &status);

  // This loop moves to the next row of matrix_x
  sli_mvp_pb_begin_loop(p, a_rows, &status); {
    //This loop moves to the next column of matrix_y
    sli_mvp_pb_begin_loop(p, b_cols, &status); {
      sli_mvp_pb_compute(p,
                         SLI_MVP_OP(CLEAR),
                         SLI_MVP_ALU_Z(SLI_MVP_R3),
                         0,
                         0,
                         &status);
      // This loop calculates the dot product of a row of matrix_x and a column of matrix_y
      sli_mvp_pb_begin_loop(p, a_cols, &status); {
        if (parallel) {
          /**
           * They key here is to scale each pair of values in matrix_y by a single value in matrix_x
           * For two equal matrices X and Y with the values:
           * 1 2
           * 3 4
           * The parallel computation will look like:
           * Z_00 = (1)*1+(2)*3
           * Z_01 = (1)*2+(2)*4
           * Notice that the values marked in parentheses are essentially scale values.
           * Scaling can be achieved by using complex MAC with X as a binary16 and Y as complex_binary16.
           **/
          sli_mvp_pb_compute(p,
                             SLI_MVP_OP(MACC),
                             SLI_MVP_ALU_X(SLI_MVP_R1) | SLI_MVP_ALU_Y(SLI_MVP_R2) | SLI_MVP_ALU_A(SLI_MVP_R3) | SLI_MVP_ALU_Z(SLI_MVP_R3),
                             SLI_MVP_LOAD(0, SLI_MVP_R1, matrix_x, SLI_MVP_INCRDIM_COL) | SLI_MVP_LOAD(1, SLI_MVP_R2, matrix_y, SLI_MVP_INCRDIM_ROW),
                             0,
                             &status);
        } else {
          sli_mvp_pb_compute(p,
                             SLI_MVP_OP(MACR2A),
                             SLI_MVP_ALU_X(SLI_MVP_R1) | SLI_MVP_ALU_Y(SLI_MVP_R2) | SLI_MVP_ALU_A(SLI_MVP_R3) | SLI_MVP_ALU_Z(SLI_MVP_R3),
                             SLI_MVP_LOAD(0, SLI_MVP_R1, matrix_x, SLI_MVP_INCRDIM_COL) | SLI_MVP_LOAD(1, SLI_MVP_R2, matrix_y, SLI_MVP_INCRDIM_ROW),
                             0,
                             &status);
        }
      }
      sli_mvp_pb_end_loop(p);

      sli_mvp_pb_postloop_incr_dim(p, matrix_y, SLI_MVP_INCRDIM_COL);
      sli_mvp_pb_compute(p,
                         SLI_MVP_OP(NOOP),
                         0,
                         0,
                         SLI_MVP_STORE(SLI_MVP_R3, matrix_z, SLI_MVP_INCRDIM_COL),
                         &status);
    }
    sli_mvp_pb_end_loop(p);

    sli_mvp_pb_postloop_incr_dim(p, matrix_x, SLI_MVP_INCRDIM_ROW);
    sli_mvp_pb_postloop_incr_dim(p, matrix_z, SLI_MVP_INCRDIM_ROW);
  }
  sli_mvp_pb_end_loop(p);

  sli_mvp_pb_execute_program(p);
  #else
  sli_mvp_power_program_prepare();

  // Program array controllers.
  MVP->ARRAY[0].DIM0CFG = SLI_MVP_DATATYPE_BINARY16 << _MVP_ARRAYDIM0CFG_BASETYPE_SHIFT;
  MVP->ARRAY[0].DIM1CFG = ((a_rows - 1) << _MVP_ARRAYDIM1CFG_SIZE_SHIFT) | (a_cols << _MVP_ARRAYDIM1CFG_STRIDE_SHIFT);
  MVP->ARRAY[0].DIM2CFG = ((a_cols - 1) << _MVP_ARRAYDIM2CFG_SIZE_SHIFT) | (1 << _MVP_ARRAYDIM2CFG_STRIDE_SHIFT);
  MVP->ARRAY[0].ADDRCFG = MVP_ARRAY_PTR(input_a->data);

  MVP->ARRAY[1].DIM0CFG = data_type << _MVP_ARRAYDIM0CFG_BASETYPE_SHIFT;
  MVP->ARRAY[1].DIM1CFG = ((b_rows - 1) << _MVP_ARRAYDIM1CFG_SIZE_SHIFT) | (b_cols << _MVP_ARRAYDIM1CFG_STRIDE_SHIFT);
  MVP->ARRAY[1].DIM2CFG = ((b_cols - 1) << _MVP_ARRAYDIM2CFG_SIZE_SHIFT) | (1 << _MVP_ARRAYDIM2CFG_STRIDE_SHIFT);
  MVP->ARRAY[1].ADDRCFG = MVP_ARRAY_PTR(input_b->data);

  MVP->ARRAY[2].DIM0CFG = data_type << _MVP_ARRAYDIM0CFG_BASETYPE_SHIFT;
  MVP->ARRAY[2].DIM1CFG = ((z_rows - 1) << _MVP_ARRAYDIM1CFG_SIZE_SHIFT) | (z_cols << _MVP_ARRAYDIM1CFG_STRIDE_SHIFT);
  MVP->ARRAY[2].DIM2CFG = ((z_cols - 1) << _MVP_ARRAYDIM2CFG_SIZE_SHIFT) | (1 << _MVP_ARRAYDIM2CFG_STRIDE_SHIFT);
  MVP->ARRAY[2].ADDRCFG = MVP_ARRAY_PTR(output->data);

  // Program loop controllers.

  MVP->LOOP[1].RST = 0;
  MVP->LOOP[2].RST = 0;

  MVP->LOOP[0].CFG = ((a_rows - 1) << _MVP_LOOPCFG_NUMITERS_SHIFT);
  MVP->LOOP[1].CFG = ((b_cols - 1) << _MVP_LOOPCFG_NUMITERS_SHIFT)
                     | ((SLI_MVP_LOOP_INCRDIM(matrix_x, SLI_MVP_INCRDIM_ROW) | SLI_MVP_LOOP_INCRDIM(matrix_z, SLI_MVP_INCRDIM_ROW)) << _MVP_LOOPCFG_ARRAY0INCRDIM0_SHIFT);
  MVP->LOOP[2].CFG = ((a_cols - 1) << _MVP_LOOPCFG_NUMITERS_SHIFT)
                     | (SLI_MVP_LOOP_INCRDIM(matrix_y, SLI_MVP_INCRDIM_COL) << _MVP_LOOPCFG_ARRAY0INCRDIM0_SHIFT);

  // Program instruction.
  MVP->INSTR[0].CFG0 = SLI_MVP_ALU_Z(SLI_MVP_R3);
  MVP->INSTR[0].CFG1 = 0;
  MVP->INSTR[0].CFG2 = (SLI_MVP_OP(CLEAR) << _MVP_INSTRCFG2_ALUOP_SHIFT)
                       | MVP_INSTRCFG2_LOOP0BEGIN
                       | MVP_INSTRCFG2_LOOP1BEGIN;

  MVP->INSTR[1].CFG0 =  SLI_MVP_ALU_X(SLI_MVP_R1) | SLI_MVP_ALU_Y(SLI_MVP_R2) | SLI_MVP_ALU_A(SLI_MVP_R3) | SLI_MVP_ALU_Z(SLI_MVP_R3);
  MVP->INSTR[1].CFG1 = SLI_MVP_LOAD(0, SLI_MVP_R1, matrix_x, SLI_MVP_INCRDIM_COL) | SLI_MVP_LOAD(1, SLI_MVP_R2, matrix_y, SLI_MVP_INCRDIM_ROW);
  if (parallel) {
    MVP->INSTR[1].CFG2 = (SLI_MVP_OP(MACC) << _MVP_INSTRCFG2_ALUOP_SHIFT)
                         | MVP_INSTRCFG2_LOOP2BEGIN
                         | MVP_INSTRCFG2_LOOP2END;
  } else {
    MVP->INSTR[1].CFG2 = (SLI_MVP_OP(MACR2A) << _MVP_INSTRCFG2_ALUOP_SHIFT)
                         | MVP_INSTRCFG2_LOOP2BEGIN
                         | MVP_INSTRCFG2_LOOP2END;
  }

  MVP->INSTR[2].CFG0 = 0;
  MVP->INSTR[2].CFG1 = SLI_MVP_STORE(SLI_MVP_R3, matrix_z, SLI_MVP_INCRDIM_COL);
  MVP->INSTR[2].CFG2 = (SLI_MVP_OP(NOOP) << _MVP_INSTRCFG2_ALUOP_SHIFT)
                       | MVP_INSTRCFG2_LOOP0END
                       | MVP_INSTRCFG2_LOOP1END
                       | MVP_INSTRCFG2_ENDPROG;

  // Start program.
  MVP->CMD = MVP_CMD_INIT | MVP_CMD_START;
  MVP_SIMULATOR_EXECUTE();
  #endif

  sli_mvp_wait_for_completion();
  return SL_STATUS_OK;
}
