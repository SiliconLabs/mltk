#pragma once

#include <cstdint>
#include <cassert>
#include <type_traits>
#include "msgpack.h"


#if __cplusplus < 201703L
#define constexpr
#endif

namespace mltk
{


/**
 * Pack a scalar into to the current dictionary
 *
 * This packs a scalar value with the corresponding dictionary key into the given @ref msgpack_context_t.
 * The @ref msgpack_context_t should have been previously initialized.
 *
 * @ref msgpack_write_dict_marker() MUST have been previously called to specify the dictionary.
 *
 * @param context Previously initialize @ref msgpack_context_t
 * @param key Dictionary key of written value
 * @param value scalar value
 * @return 0 on success
 */
template<typename T>
int msgpack_write_dict(msgpack_context_t* context, const char* key, T v)
{
    if constexpr(std::is_same<T, bool>::value)
    {
        return msgpack_write_dict_bool(context, key, v);
    }
    else if constexpr(std::is_same<T, uint64_t>::value)
    {
        return msgpack_write_dict_ulong(context, key, v);
    }
    else if constexpr(std::is_same<T, uint32_t>::value)
    {
        return msgpack_write_dict_uint(context, key, v);
    }
    else if constexpr(std::is_same<T, int64_t>::value)
    {
        return msgpack_write_dict_long(context, key, v);
    }
    else if constexpr(std::is_same<T, int32_t>::value)
    {
        return msgpack_write_dict_int(context, key, v);
    }
    else if constexpr(std::is_same<T, double>::value)
    {
        return msgpack_write_dict_double(context, key, v);
    }
    else if constexpr(std::is_same<T, float>::value)
    {
        return msgpack_write_dict_float(context, key, v);
    }
    else if constexpr(std::is_same<T, char*>::value || std::is_same<T, const char*>::value)
    {
        if(v == nullptr)
        {
            return msgpack_write_dict_nil(context, key);
        }
        return msgpack_write_dict_str(context, key, v);
    }
    else 
    {
        assert("Unsupported data type");
        return -1;
    }
}


/**
 * Pack a scalar value
 *
 * This packs a scalar value into the given @ref msgpack_context_t.
 * The @ref msgpack_context_t should have been previously initialized.
 *
 * @param context Previously initialize @ref msgpack_context_t
 * @param value Scalar value to write
 * @return 0 on success
 */
template<typename T>
int msgpack_write(msgpack_context_t* context, T v)
{
    if(std::is_same<T, bool>::value)
    {
        return msgpack_write_bool(context, v);
    }
    else if(std::is_same<T, uint64_t>::value)
    {
        return msgpack_write_ulong(context, v);
    }
    else if(std::is_same<T, uint32_t>::value)
    {
        return msgpack_write_uint(context, v);
    }
    else if(std::is_same<T, int64_t>::value)
    {
        return msgpack_write_long(context, v);
    }
    else if(std::is_same<T, int32_t>::value)
    {
        return msgpack_write_int(context, v);
    }
    else if(std::is_same<T, double>::value)
    {
        return msgpack_write_double(context, v);
    }
    else if(std::is_same<T, float>::value)
    {
        return msgpack_write_float(context, v);
    }
    else if(std::is_same<T, char*>::value || std::is_same<T, const char*>::value)
    {
        if(v == nullptr)
        {
            return msgpack_write_nil(context, v);
        }
        return msgpack_write_str(context, v);
    }
    else 
    {
        assert("Unsupported data type");
        return -1;
    }
}



} // namespace mltk