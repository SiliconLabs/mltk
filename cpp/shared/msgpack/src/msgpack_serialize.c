#include "msgpack_internal.h"


static int push_container(msgpack_context_t *context, int32_t count);
static int write_marker(msgpack_context_t *context, msgpack_marker_t marker);
static int write_marker_and_bytes(msgpack_context_t *context, msgpack_marker_t marker, const uint8_t *buffer, uint32_t length);
static int write_bytes(msgpack_context_t *context, const uint8_t *buffer, uint32_t length);




/*************************************************************************************************/
int msgpack_write_int(msgpack_context_t *context, int32_t value)
{
    uint8_t length;
    msgpack_marker_t marker;

    if(value >= 0)
    {
        return msgpack_write_uint(context, (uint32_t)value);
    }
    else if(value >= -32)
    {
        return write_marker(context, (msgpack_marker_t)value);
    }
    else if(value >= INT8_MIN)
    {
        length = sizeof(uint8_t);
        marker = S8_MARKER;
    }
    else if(value >= INT16_MIN)
    {
        length = sizeof(uint16_t);
        marker = S16_MARKER;
    }
    else
    {
        length = sizeof(uint32_t);
        marker = S32_MARKER;
    }

    return write_marker_and_bytes(context, marker, (const uint8_t*)&value, length);
}

/*************************************************************************************************/
int msgpack_write_long(msgpack_context_t *context, int64_t value)
{
    int result;

    if(value >= 0)
    {
        result = msgpack_write_ulong(context, (uint64_t)value);
    }
    else if(value >= INT32_MIN)
    {
        result = msgpack_write_int(context, (int32_t)value);
    }
    else
    {
        result = write_marker_and_bytes(context, S64_MARKER, (const uint8_t*)&value, sizeof(uint64_t));
    }

    return result;
}

/*************************************************************************************************/
int msgpack_write_uint(msgpack_context_t *context, uint32_t value)
{
    uint8_t length;
    msgpack_marker_t marker;

    if(value <= INT8_MAX)
    {
        return write_marker(context, (msgpack_marker_t)value);
    }
    else if(value <= UINT8_MAX)
    {
        length = sizeof(uint8_t);
        marker = U8_MARKER;
    }
    else if(value <= UINT16_MAX)
    {
        length = sizeof(uint16_t);
        marker = U16_MARKER;
    }
    else
    {
        length = sizeof(uint32_t);
        marker = U32_MARKER;
    }

    return write_marker_and_bytes(context, marker, (const uint8_t*)&value, length);
}

/*************************************************************************************************/
int msgpack_write_ulong(msgpack_context_t *context, uint64_t value)
{
    int result;

    if(value <= UINT32_MAX)
    {
        result = msgpack_write_uint(context, (uint32_t)value);
    }
    else
    {
        result = write_marker_and_bytes(context, U64_MARKER, (const uint8_t*)&value, sizeof(uint64_t));
    }

    return result;
}

/*************************************************************************************************/
int msgpack_write_float(msgpack_context_t *context, float value)
{
    return write_marker_and_bytes(context, FLOAT_MARKER, (const uint8_t*)&value, sizeof(uint32_t));
}

/*************************************************************************************************/
int msgpack_write_double(msgpack_context_t *context, double value)
{
    return write_marker_and_bytes(context, DOUBLE_MARKER, (const uint8_t*)&value, sizeof(uint64_t));
}

/*************************************************************************************************/
int msgpack_write_nil(msgpack_context_t *context)
{
    return write_marker(context, NIL_MARKER);
}

/*************************************************************************************************/
int msgpack_write_bool(msgpack_context_t *context, bool value)
{
    msgpack_marker_t b = value ? TRUE_MARKER : FALSE_MARKER;
    return write_marker(context, b);
}

/*************************************************************************************************/
int msgpack_write_str_marker(msgpack_context_t *context, uint32_t size)
{
    if(size <= FIXSTR_SIZE)
    {
        const msgpack_marker_t marker = FIXSTR_MARKER | size;
        return write_marker(context, marker);
    }
    else
    {
        const uint8_t marker = (size <= UINT8_MAX)      ? STR8_MARKER :
                               (size <= UINT16_MAX)     ? STR16_MARKER : STR32_MARKER;
        const uint8_t length = (size <= UINT8_MAX)      ? sizeof(uint8_t) :
                               (size <= UINT16_MAX)     ? sizeof(uint16_t) : sizeof(uint32_t);
        return write_marker_and_bytes(context, marker, (const uint8_t*)&size, length);
    }
}

/*************************************************************************************************/
int msgpack_write_str(msgpack_context_t *context, const char *str)
{
    int result;
    const uint32_t len = strlen(str);

    if(CHECK_FAILURE(result, msgpack_write_str_marker(context, len)))
    {
    }
    else if(CHECK_FAILURE(result, write_bytes(context, (const uint8_t*)str, len)))
    {
    }
    return result;
}

/*************************************************************************************************/
int msgpack_write_bin_marker(msgpack_context_t *context, uint32_t size)
{
    const uint8_t marker = (size <= UINT8_MAX)      ? BIN8_MARKER :
                           (size <= UINT16_MAX)     ? BIN16_MARKER : BIN32_MARKER;
    const uint8_t length = (size <= UINT8_MAX)      ? sizeof(uint8_t) :
                           (size <= UINT16_MAX)     ? sizeof(uint16_t) : sizeof(uint32_t);
    return write_marker_and_bytes(context, marker, (const uint8_t*)&size, length);
}

/*************************************************************************************************/
int msgpack_write_bin(msgpack_context_t *context, const void *data, uint32_t length)
{
    int result;

    if(CHECK_FAILURE(result, msgpack_write_bin_marker(context, length)))
    {
    }
    else if(CHECK_FAILURE(result, write_bytes(context, (const uint8_t*)data, length)))
    {
    }
    return result;
}

/*************************************************************************************************/
int msgpack_write_context(msgpack_context_t *context, const msgpack_context_t *value_context)
{
    if(value_context->buffer.ptr == NULL || value_context->buffer.buffer == NULL)
    {
        return -1;
    }

    return write_bytes(context, value_context->buffer.buffer, MSGPACK_BUFFER_USED(value_context));
}

/*************************************************************************************************/
int msgpack_write_dict_marker(msgpack_context_t *context, int32_t size)
{
#ifndef MSGPACK_MAX_NESTED_CONTAINERS
    if(size < 0)
    {
        return -1;
    }
#endif

    if((size >= 0) && !(context->flags & MSGPACK_PACK_16BIT_DICTS) && (size <= FIXMAP_SIZE))
    {
        const uint8_t marker = FIXMAP_MARKER | size;
        RETURN_ON_FAILURE(write_marker(context, marker));
        RETURN_ON_FAILURE(push_container(context, size*2));
        return 0;
    }
    else
    {
        const int32_t normalized_sized = (size < 0)             ? UINT16_MAX : size;
        const uint8_t marker = (normalized_sized <= UINT16_MAX) ? MAP16_MARKER : MAP32_MARKER;
        const uint8_t length = (normalized_sized <= UINT16_MAX) ? sizeof(uint16_t) : sizeof(uint32_t);

        RETURN_ON_FAILURE(write_marker_and_bytes(context, marker, (const uint8_t*)&size, length));
        RETURN_ON_FAILURE(push_container(context, size*2));
        return 0;
    }
}

/*************************************************************************************************/
int msgpack_write_array_marker(msgpack_context_t *context, int32_t size)
{
#ifndef MSGPACK_MAX_NESTED_CONTAINERS
    if(size < 0)
    {
        return -1;
    }
#endif

    if((size >= 0) && size <= FIXARRAY_SIZE)
    {
        const uint8_t marker = FIXARRAY_MARKER | size;
        RETURN_ON_FAILURE(write_marker(context, marker));
        RETURN_ON_FAILURE(push_container(context, size));
        return 0;
    }
    else
    {
        const int32_t normalized_sized = (size < 0)             ? UINT16_MAX : size;
        const uint8_t marker = (normalized_sized <= UINT16_MAX) ? ARRAY16_MARKER : ARRAY32_MARKER;
        const uint8_t length = (normalized_sized <= UINT16_MAX) ? sizeof(uint16_t) : sizeof(uint32_t);

        RETURN_ON_FAILURE(write_marker_and_bytes(context, marker, (const uint8_t*)&size, length));
        RETURN_ON_FAILURE(push_container(context, size));
        return 0;
    }
}

/*************************************************************************************************/
int msgpack_write_dict_nil(msgpack_context_t *context, const char*key)
{
    int result;

    if(CHECK_FAILURE(result, msgpack_write_str(context, key)))
    {
    }
    else if(CHECK_FAILURE(result, msgpack_write_nil(context)))
    {
    }

    return result;
}

/*************************************************************************************************/
int msgpack_write_dict_bool(msgpack_context_t *context, const char*key, bool value)
{
    int result;

    if(CHECK_FAILURE(result, msgpack_write_str(context, key)))
    {
    }
    else if(CHECK_FAILURE(result, msgpack_write_bool(context, value)))
    {
    }

    return result;
}

/*************************************************************************************************/
int msgpack_write_dict_int(msgpack_context_t *context, const char*key, int32_t value)
{
    int result;

    if(CHECK_FAILURE(result, msgpack_write_str(context, key)))
    {
    }
    else if(CHECK_FAILURE(result, msgpack_write_int(context, value)))
    {
    }

    return result;
}

/*************************************************************************************************/
int msgpack_write_dict_uint(msgpack_context_t *context, const char*key, uint32_t value)
{
    int result;

    if(CHECK_FAILURE(result, msgpack_write_str(context, key)))
    {
    }
    else if(CHECK_FAILURE(result, msgpack_write_uint(context, value)))
    {
    }

    return result;
}

/*************************************************************************************************/
int msgpack_write_dict_long(msgpack_context_t *context, const char*key, int64_t value)
{
    int result;

    if(CHECK_FAILURE(result, msgpack_write_str(context, key)))
    {
    }
    else if(CHECK_FAILURE(result, msgpack_write_long(context, value)))
    {
    }

    return result;
}

/*************************************************************************************************/
int msgpack_write_dict_ulong(msgpack_context_t *context, const char*key, uint64_t value)
{
    int result;

    if(CHECK_FAILURE(result, msgpack_write_str(context, key)))
    {
    }
    else if(CHECK_FAILURE(result, msgpack_write_ulong(context, value)))
    {
    }

    return result;
}

/*************************************************************************************************/
int msgpack_write_dict_float(msgpack_context_t *context, const char*key, float value)
{
    int result;

    if(CHECK_FAILURE(result, msgpack_write_str(context, key)))
    {
    }
    else if(CHECK_FAILURE(result, msgpack_write_float(context, value)))
    {
    }

    return result;
}

/*************************************************************************************************/
int msgpack_write_dict_double(msgpack_context_t *context, const char*key, double value)
{
    int result;

    if(CHECK_FAILURE(result, msgpack_write_str(context, key)))
    {
    }
    else if(CHECK_FAILURE(result, msgpack_write_double(context, value)))
    {
    }

    return result;
}

/*************************************************************************************************/
int msgpack_write_dict_str(msgpack_context_t *context, const char*key, const char *value)
{
    int result;

    if(key == NULL || value == NULL)
    {
        result = -1;
    }
    else if(CHECK_FAILURE(result, msgpack_write_str(context, key)))
    {
    }
    else if(CHECK_FAILURE(result, msgpack_write_str(context, value)))
    {
    }

    return result;
}

/*************************************************************************************************/
int msgpack_write_dict_bin(msgpack_context_t *context, const char*key, const void *value, uint32_t length)
{
    int result;

    if(CHECK_FAILURE(result, msgpack_write_str(context, key)))
    {
    }
    else if(CHECK_FAILURE(result, msgpack_write_bin(context, value, length)))
    {
    }

    return result;
}

/*************************************************************************************************/
int msgpack_write_dict_dict(msgpack_context_t *context, const char*key, int32_t dict_count)
{
    int result;

    if(CHECK_FAILURE(result, msgpack_write_str(context, key)))
    {
    }
    else if(CHECK_FAILURE(result, msgpack_write_dict_marker(context, dict_count)))
    {
    }

    return result;
}

/*************************************************************************************************/
int msgpack_write_dict_array(msgpack_context_t *context, const char*key, int32_t array_count)
{
    int result;

    if(CHECK_FAILURE(result, msgpack_write_str(context, key)))
    {
    }
    else if(CHECK_FAILURE(result, msgpack_write_array_marker(context, array_count)))
    {
    }

    return result;
}

/*************************************************************************************************/
int msgpack_write_dict_context(msgpack_context_t *context, const char*key, const msgpack_context_t *value_context)
{
    int result;

    if(CHECK_FAILURE(result, msgpack_write_str(context, key)))
    {
    }
    else if(CHECK_FAILURE(result, msgpack_write_context(context, value_context)))
    {
    }

    return result;
}


/*************************************************************************************************/
int msgpack_finalize_dynamic(msgpack_context_t *context)
{
#ifdef MSGPACK_MAX_NESTED_CONTAINERS
    if(context == NULL || \
        context->container_index < 0 || \
        context->containers[context->container_index].marker_offset == UINT32_MAX || \
        context->buffer.buffer == NULL)
    {
        return -1;
    }

    uint32_t count = context->containers[context->container_index].count;
    const uint32_t marker_offset = context->containers[context->container_index].marker_offset;

    uint8_t* marker_ptr = &context->buffer.buffer[marker_offset];

    // The header should always start with a 16bit map or 16bitarray
    if(!(marker_ptr[0] == MAP16_MARKER || marker_ptr[0] == ARRAY16_MARKER))
    {
        // This API currently only supports updating 16bit dicts
        return -1;
    }

    if(marker_ptr[0] == MAP16_MARKER)
    {
        count /= 2;
    }

    // Write the updated count
    marker_ptr[1] = (uint8_t)(count >> 8);
    marker_ptr[2] = (uint8_t)(count);

    context->containers[context->container_index].count = 0;
    context->containers[context->container_index].marker_offset = UINT32_MAX;
    context->container_index -= 1;

    return 0;
#else 
    return -1;
#endif
}



/** --------------------------------------------------------------------------------------------
 *  Internal functions
 * -------------------------------------------------------------------------------------------- **/



/*************************************************************************************************/
static int push_container(msgpack_context_t *context, int32_t count)
{
#ifdef MSGPACK_MAX_NESTED_CONTAINERS
    if(context == NULL)
    {
        return -1;
    }
    else if(count == 0)
    {
        return 0;
    }
    
    context->container_index += 1;

    if(context->container_index >= MSGPACK_MAX_NESTED_CONTAINERS)
    {
        return -1;
    }

    context->containers[context->container_index].count = (count < 0) ? 0 : count;

    if(count < 0)
    {
        if(context->buffer.ptr != NULL && context->buffer.buffer != NULL)
        {
            context->containers[context->container_index].marker_offset = (uintptr_t)(context->buffer.ptr - context->buffer.buffer) - 3;
        }
        else
        {
            return -1;
        }
    }
    else 
    {
        context->containers[context->container_index].marker_offset = UINT32_MAX;
    }
#endif

    return 0;
}

/*************************************************************************************************/
static int write_marker(msgpack_context_t *context, msgpack_marker_t marker)
{
    RETURN_ON_FAILURE(write_bytes(context, &marker, 1));

#ifdef MSGPACK_MAX_NESTED_CONTAINERS
    if(context->container_index >= 0)
    {
        if(context->containers[context->container_index].marker_offset == UINT32_MAX)
        {
            if(context->containers[context->container_index].count == 0)
            {
                return -1;
            }

            context->containers[context->container_index].count -= 1;
            if(context->containers[context->container_index].count == 0)
            {
                if(context->container_index < 0)
                {
                    return -1;
                }
                context->container_index -= 1;
            }
        }
        else 
        {
            context->containers[context->container_index].count += 1;
        }
    }
#endif // MSGPACK_MAX_NESTED_CONTAINERS

    return 0;
}


/*************************************************************************************************/
static int write_marker_and_bytes(msgpack_context_t *context, msgpack_marker_t marker, const uint8_t *buffer, uint32_t length)
{
    int result;

    if(!CHECK_FAILURE(result, write_marker(context, marker)))
    {
        uint8_t converted_buffer[sizeof(uint64_t)];

        HTON_BUFFER(buffer, converted_buffer, length);

        if(CHECK_FAILURE(result, write_bytes(context, converted_buffer, length)))
        {
        }
    }

    return result;
}

/*************************************************************************************************/
static int write_bytes(msgpack_context_t *context, const uint8_t *buffer, uint32_t length)
{
    if(context == NULL)
    {
        return -1;
    }
    else if(context->writer != NULL)
    {
        return context->writer(context->user, buffer, length);
    }
    else if(context->buffer.buffer != NULL)
    {
        if(context->buffer.ptr == NULL)
        {
            return -1;
        }
        else if(context->buffer.ptr + length > context->buffer.end)
        {
            return -1;
        }

        memcpy(context->buffer.ptr, buffer, length);
        context->buffer.ptr += length;

        return 0;
    }
    else
    {
        return -1;
    }
}