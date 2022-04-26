#pragma once

#include <cstdint>


namespace cpputils
{

/**
 * Get a pseudo-random number.
 *
 * @param[in] seed: Seed value of pseudo-random number generator.
 * This value takes a pointer to 32 bit word. The seed tracks the current
 * position in the pseudo-random sequence and its state will be updated with
 * each call to pseudo_rand. The application should not use the seed value
 * itself as a source of pseudo random numbers.
 *
 * @return 32 bit word from pseudo-random sequence.
 */
uint32_t pseudo_rand(uint32_t seed = 0);



} // namespace cpputils

