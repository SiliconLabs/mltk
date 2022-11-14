#pragma once

#include <stdbool.h>
#include <stdio.h>

#include "msgpack.h"


#define HTON_BUFFER(src, dst, length) convert_endianness((const uint8_t*)src, (uint8_t*)dst, length)
#define CHECK_FAILURE(result, func) ((result = (func)) != 0)
#define RETURN_ON_FAILURE(func) { int _result = (func); if(_result != 0) return _result; }

#ifndef MIN
#define MIN(x,y)  ((x) < (y) ? (x) : (y))
#endif /* ifndef MIN */
#ifndef MAX
#define MAX(x,y)  ((x) > (y) ? (x) : (y))
#endif /* ifndef MAX */




typedef uint8_t msgpack_marker_t;
enum msgpack_marker_enum
{
    POSITIVE_FIXNUM_MARKER = 0x00,
    FIXMAP_MARKER          = 0x80,
    FIXARRAY_MARKER        = 0x90,
    FIXSTR_MARKER          = 0xA0,
    NIL_MARKER             = 0xC0,
    FALSE_MARKER           = 0xC2,
    TRUE_MARKER            = 0xC3,
    BIN8_MARKER            = 0xC4,
    BIN16_MARKER           = 0xC5,
    BIN32_MARKER           = 0xC6,
    EXT8_MARKER            = 0xC7,
    EXT16_MARKER           = 0xC8,
    EXT32_MARKER           = 0xC9,
    FLOAT_MARKER           = 0xCA,
    DOUBLE_MARKER          = 0xCB,
    U8_MARKER              = 0xCC,
    U16_MARKER             = 0xCD,
    U32_MARKER             = 0xCE,
    U64_MARKER             = 0xCF,
    S8_MARKER              = 0xD0,
    S16_MARKER             = 0xD1,
    S32_MARKER             = 0xD2,
    S64_MARKER             = 0xD3,
    FIXEXT1_MARKER         = 0xD4,
    FIXEXT2_MARKER         = 0xD5,
    FIXEXT4_MARKER         = 0xD6,
    FIXEXT8_MARKER         = 0xD7,
    FIXEXT16_MARKER        = 0xD8,
    STR8_MARKER            = 0xD9,
    STR16_MARKER           = 0xDA,
    STR32_MARKER           = 0xDB,
    ARRAY16_MARKER         = 0xDC,
    ARRAY32_MARKER         = 0xDD,
    MAP16_MARKER           = 0xDE,
    MAP32_MARKER           = 0xDF,
    NEGATIVE_FIXNUM_MARKER = 0xE0
};

typedef uint8_t msgpack_size_t;
enum msgpack_size_enum
{
    FIXARRAY_SIZE        = 0xF,
    FIXMAP_SIZE          = 0xF,
    FIXSTR_SIZE          = 0x1F
};


typedef struct
{
    void *context;
} msgpack_user_context_t;




/*************************************************************************************************/
static inline void convert_endianness(const uint8_t *src, uint8_t *dst, uint8_t length)
{
    const uint8_t *src_ptr = &src[length-1];
    while(length-- > 0)
    {
        *dst++ = *src_ptr--;
    }
}