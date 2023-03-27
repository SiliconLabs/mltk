/***************************************************************************//**
 * @file
 * @brief MVP Math vector offset functions.
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
#include "sl_mvp_power.h"
#include "sl_mvp_util.h"
#include "sl_mvp_program_area.h"
#include "sl_math_mvp_vector_offset.h"

#ifndef USE_MVP_PROGRAMBUILDER
#define USE_MVP_PROGRAMBUILDER    0
#endif

sl_status_t sl_math_mvp_vector_offset_f16(const float16_t *input,
                                          const float16_t offset,
                                          float16_t *output,
                                          size_t num_elements)
{
  sl_status_t status = SL_STATUS_OK;
  sli_mvp_datatype_t data_type;
  size_t len, remainder;
  size_t i, parallel;
  uint32_t rows, cols;

  if (!input || !output || !num_elements) {
    return SL_STATUS_INVALID_PARAMETER;
  }

  len = num_elements;
  parallel = 1;
  remainder = 0;

  data_type = SLI_MVP_DATATYPE_BINARY16;
  if (sli_mvp_util_is_pointer_word_aligned((float16_t *)input)
      && sli_mvp_util_is_pointer_word_aligned(output)
      && (len >= 2U)) {
    parallel = 2U;
    if (len & 1U ) {
      remainder++;
    }
    data_type = SLI_MVP_DATATYPE_COMPLEX_BINARY16;
    len /= 2U;
  }

  if (len <= SLI_MVP_MAX_ROW_LENGTH) {
    rows = 1;
    cols = len;
  } else {
    i = len;
    while ((status = sli_mvp_util_factorize_number(i, 1024U, &rows, &cols)) != SL_STATUS_OK) {
      i--;
      remainder += parallel;
    }
  }

#if USE_MVP_PROGRAMBUILDER

  const int vector_x = SLI_MVP_ARRAY(0);
  const int vector_z = SLI_MVP_ARRAY(1);

  sli_mvp_program_context_t *p = sli_mvp_get_program_area_context();
  sli_mvp_pb_init_program(p);
  sli_mvp_pb_begin_program(p);

  sli_mvp_pb_config_matrix(p->p, vector_x, (void *)input, data_type, rows, cols, &status);
  sli_mvp_pb_config_matrix(p->p, vector_z, output, data_type, rows, cols, &status);

  sli_mvp_prog_set_reg_f16c(p->p, SLI_MVP_R1, offset, offset);
  sli_mvp_prog_set_reg_f16c(p->p, SLI_MVP_R2, float16_t(0), float16_t(0));

  sli_mvp_pb_begin_loop(p, rows, &status); {
    sli_mvp_pb_begin_loop(p, cols, &status); {
      sli_mvp_pb_compute(p,
                         SLI_MVP_OP(AACC),
                         SLI_MVP_ALU_X(SLI_MVP_R0)
                         | SLI_MVP_ALU_Y(SLI_MVP_R1)
                         | SLI_MVP_ALU_A(SLI_MVP_R2)
                         | SLI_MVP_ALU_Z(SLI_MVP_R3),
                         SLI_MVP_LOAD(0, SLI_MVP_R0, vector_x, SLI_MVP_INCRDIM_WIDTH),
                         SLI_MVP_STORE(SLI_MVP_R3, vector_z, SLI_MVP_INCRDIM_WIDTH),
                         &status);
    }
    sli_mvp_pb_end_loop(p);
    sli_mvp_pb_postloop_incr_dim(p, vector_x, SLI_MVP_INCRDIM_HEIGHT);
    sli_mvp_pb_postloop_incr_dim(p, vector_z, SLI_MVP_INCRDIM_HEIGHT);
  }
  sli_mvp_pb_end_loop(p);

  // Check if any errors found during program generation.
  if (status != SL_STATUS_OK) {
    return status;
  }
  sli_mvp_pb_execute_program(p);

#else

  sli_mvp_power_program_prepare();

  // Program array controllers.
  MVP->ARRAY[0].DIM0CFG = MVP->ARRAY[1].DIM0CFG =
    data_type << _MVP_ARRAYDIM0CFG_BASETYPE_SHIFT;
  MVP->ARRAY[0].DIM1CFG = MVP->ARRAY[1].DIM1CFG =
    ((rows - 1) << _MVP_ARRAYDIM1CFG_SIZE_SHIFT) | (cols << _MVP_ARRAYDIM1CFG_STRIDE_SHIFT);
  MVP->ARRAY[0].DIM2CFG = MVP->ARRAY[1].DIM2CFG =
    ((cols - 1) << _MVP_ARRAYDIM2CFG_SIZE_SHIFT) | (1 << _MVP_ARRAYDIM2CFG_STRIDE_SHIFT);
  MVP->ARRAY[0].ADDRCFG = MVP_ARRAY_PTR(input);
  MVP->ARRAY[1].ADDRCFG = MVP_ARRAY_PTR(output);

  // Program registers
  sli_mvp_alu_reg_t r;
  r.reg.value.real = offset;
  r.reg.value.imag = offset;
  MVP->ALU[1].REGSTATE = r.REGSTATE;
  MVP->ALU[2].REGSTATE = 0;

  // Program loop controllers.
  MVP->LOOP[1].RST = 0;
  MVP->LOOP[0].CFG = (rows - 1) << _MVP_LOOPCFG_NUMITERS_SHIFT;
  MVP->LOOP[1].CFG = ((cols - 1) << _MVP_LOOPCFG_NUMITERS_SHIFT)
                     | ((SLI_MVP_LOOP_INCRDIM(SLI_MVP_ARRAY(0), SLI_MVP_INCRDIM_HEIGHT)
                         | SLI_MVP_LOOP_INCRDIM(SLI_MVP_ARRAY(1), SLI_MVP_INCRDIM_HEIGHT))
                        << _MVP_LOOPCFG_ARRAY0INCRDIM0_SHIFT);

  // Program instruction.
  MVP->INSTR[0].CFG0 = SLI_MVP_ALU_X(SLI_MVP_R0)
                       | SLI_MVP_ALU_Y(SLI_MVP_R1)
                       | SLI_MVP_ALU_A(SLI_MVP_R2)
                       | SLI_MVP_ALU_Z(SLI_MVP_R3);
  MVP->INSTR[0].CFG1 = SLI_MVP_LOAD(0, SLI_MVP_R0, SLI_MVP_ARRAY(0), SLI_MVP_INCRDIM_WIDTH)
                       | SLI_MVP_STORE(SLI_MVP_R3, SLI_MVP_ARRAY(1), SLI_MVP_INCRDIM_WIDTH);
  MVP->INSTR[0].CFG2 = (SLI_MVP_OP(AACC) << _MVP_INSTRCFG2_ALUOP_SHIFT)
                       | MVP_INSTRCFG2_ENDPROG
                       | MVP_INSTRCFG2_LOOP0BEGIN
                       | MVP_INSTRCFG2_LOOP0END
                       | MVP_INSTRCFG2_LOOP1BEGIN
                       | MVP_INSTRCFG2_LOOP1END;

  // Start program.
  MVP->CMD = MVP_CMD_INIT | MVP_CMD_START;
  MVP_SIMULATOR_EXECUTE();

#endif // USE_MVP_PROGRAMBUILDER

  // When factorization above is incomplete, handle the reminders here.
  i = num_elements - remainder;
  while (remainder--) {
    output[i] = input[i] + offset;
    i++;
  }

  sli_mvp_wait_for_completion();

  return SL_STATUS_OK;
}
