#pragma once 


#ifdef __arm__

typedef __fp16 float16_t;

#else

#include "half.hpp"

typedef half_float::half float16_t;

#endif