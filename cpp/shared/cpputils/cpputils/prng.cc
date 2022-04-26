/*******************************************************************************
 * # License
 * Copyright 2019 Silicon Laboratories Inc. www.silabs.com
 *******************************************************************************
 *
 * The licensor of this software is Silicon Laboratories Inc. Your use of this
 * software is governed by the terms of Silicon Labs Master Software License
 * Agreement (MSLA) available at
 * www.silabs.com/about-us/legal/master-software-license-agreement. This
 * software is distributed to you in Source Code format and is governed by the
 * sections of the MSLA applicable to Source Code.
 *
 ******************************************************************************/

#include "cpputils/prng.hpp"

/*

This is a simple LFSR based PRNG, loosely based on Maxim's App Note 4400.
https://www.maximintegrated.com/en/app-notes/index.mvp/id/4400

However, we make a number of changes:

1) We only use a single 32 bit LFSR, instead cascading with the 31 bit.
2) We extract only 1 bit per shift, so we need to run the lfsr 32 times
to get a 32 bit word. This is how LFSRs are supposed to work.  Maxim's code
is faster but doesn't produce a high quality sequence (though it may be
fine when cascaded).
*/


namespace cpputils
{


#define LFSR_POLY_32 0xB4BCD35C
#define LFSR_POLY_31 0x7A5BC2E3 // unused

static uint32_t lfsr(uint32_t *value, uint32_t poly)
{
    uint32_t feedback = *value & 1;

    *value >>= 1;
    if(feedback)
    {
        *value ^= poly;
    }

    return *value;
}


uint32_t pseudo_rand(uint32_t seed)
{
    static uint32_t seed_buffer = 123456789;

    if(seed != 0)
    {
        seed_buffer = seed;
    }

    // Ensure seed is not 0, the LFSR will not work with all 0s.
    // if it is 0, set it to 1.  This ensures the LFSR does not
    // get stuck and is deterministic.
    if(seed_buffer == 0)
    {
        seed_buffer = 1;
    }

    uint32_t value = (lfsr(&seed_buffer, LFSR_POLY_32) & 1) << 31;
    value |= (lfsr(&seed_buffer, LFSR_POLY_32) & 1) << 30;
    value |= (lfsr(&seed_buffer, LFSR_POLY_32) & 1) << 29;
    value |= (lfsr(&seed_buffer, LFSR_POLY_32) & 1) << 28;
    value |= (lfsr(&seed_buffer, LFSR_POLY_32) & 1) << 27;
    value |= (lfsr(&seed_buffer, LFSR_POLY_32) & 1) << 26;
    value |= (lfsr(&seed_buffer, LFSR_POLY_32) & 1) << 25;
    value |= (lfsr(&seed_buffer, LFSR_POLY_32) & 1) << 24;
    value |= (lfsr(&seed_buffer, LFSR_POLY_32) & 1) << 23;
    value |= (lfsr(&seed_buffer, LFSR_POLY_32) & 1) << 22;
    value |= (lfsr(&seed_buffer, LFSR_POLY_32) & 1) << 21;
    value |= (lfsr(&seed_buffer, LFSR_POLY_32) & 1) << 20;
    value |= (lfsr(&seed_buffer, LFSR_POLY_32) & 1) << 19;
    value |= (lfsr(&seed_buffer, LFSR_POLY_32) & 1) << 18;
    value |= (lfsr(&seed_buffer, LFSR_POLY_32) & 1) << 17;
    value |= (lfsr(&seed_buffer, LFSR_POLY_32) & 1) << 16;
    value |= (lfsr(&seed_buffer, LFSR_POLY_32) & 1) << 15;
    value |= (lfsr(&seed_buffer, LFSR_POLY_32) & 1) << 14;
    value |= (lfsr(&seed_buffer, LFSR_POLY_32) & 1) << 13;
    value |= (lfsr(&seed_buffer, LFSR_POLY_32) & 1) << 12;
    value |= (lfsr(&seed_buffer, LFSR_POLY_32) & 1) << 11;
    value |= (lfsr(&seed_buffer, LFSR_POLY_32) & 1) << 10;
    value |= (lfsr(&seed_buffer, LFSR_POLY_32) & 1) << 9;
    value |= (lfsr(&seed_buffer, LFSR_POLY_32) & 1) << 8;
    value |= (lfsr(&seed_buffer, LFSR_POLY_32) & 1) << 7;
    value |= (lfsr(&seed_buffer, LFSR_POLY_32) & 1) << 6;
    value |= (lfsr(&seed_buffer, LFSR_POLY_32) & 1) << 5;
    value |= (lfsr(&seed_buffer, LFSR_POLY_32) & 1) << 4;
    value |= (lfsr(&seed_buffer, LFSR_POLY_32) & 1) << 3;
    value |= (lfsr(&seed_buffer, LFSR_POLY_32) & 1) << 2;
    value |= (lfsr(&seed_buffer, LFSR_POLY_32) & 1) << 1;
    value |= (lfsr(&seed_buffer, LFSR_POLY_32) & 1) << 0;

    return value;
}


} // namespace cpputils
