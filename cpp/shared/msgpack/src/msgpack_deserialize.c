#include "msgpack_internal.h"


static int deserialize_dict(msgpack_context_t *context, msgpack_object_dict_t *dict);
static int deserialize_array(msgpack_context_t *context, msgpack_object_array_t *array);
static int deserialize_next_object(msgpack_context_t *context, msgpack_object_t **obj_ptr);
static int read_and_convert_endian(msgpack_context_t *context, uint8_t *data, uint32_t length);
static int read_bytes(msgpack_context_t *context, uint8_t *data, uint32_t length);
static int skip_bytes(msgpack_context_t *context, uint32_t length);




/*************************************************************************************************/
int msgpack_deserialize_with_buffer(msgpack_object_t **root_ptr, const void *buffer, uint32_t length, msgpack_flag_t flags)
{
    int result = 0;
    msgpack_context_t context = msgpack_init_with_buffer((uint8_t*)buffer, length);
    context.flags = flags;

    *root_ptr = NULL;

    if(CHECK_FAILURE(result, deserialize_next_object(&context, root_ptr)))
    {
    }
    else if(MSGPACK_IS_DICT(*root_ptr))
    {
        result = deserialize_dict(&context, (msgpack_object_dict_t*)*root_ptr);
    }
    else if(MSGPACK_IS_ARRAY(*root_ptr))
    {
        result = deserialize_array(&context, (msgpack_object_array_t*)*root_ptr);
    }
    else
    {
        // all other objects must be encapsulated in a DICT or ARRAY
        result = -1;
    }


    if(result != 0)
    {
        msgpack_free_objects(*root_ptr);
        *root_ptr = NULL;
    }

    return result;
}



/*************************************************************************************************/
static int deserialize_dict(msgpack_context_t *context, msgpack_object_dict_t *dict)
{
    int result = 0;
    msgpack_object_t *key_obj;
    msgpack_object_t *value_obj;

    for(int i = 0; i < dict->count; ++i)
    {
        msgpack_dict_entry_t *entry = (msgpack_dict_entry_t*)&dict->entries[i];

        key_obj = NULL;
        value_obj = NULL;

        if(CHECK_FAILURE(result, deserialize_next_object(context, &key_obj)))
        {
            break;
        }
        else if(CHECK_FAILURE(result, deserialize_next_object(context, &value_obj)))
        {
            break;
        }
        else
        {
            msgpack_object_t *tmp = value_obj;
            entry->key = key_obj;
            entry->value = value_obj;

            key_obj = NULL;
            value_obj = NULL;

            if(MSGPACK_IS_DICT(tmp) &&
               CHECK_FAILURE(result, deserialize_dict(context, (msgpack_object_dict_t*)tmp)))
            {
                break;
            }
            else if(MSGPACK_IS_ARRAY(tmp) &&
                    CHECK_FAILURE(result, deserialize_array(context, (msgpack_object_array_t*)tmp)))
            {
                break;
            }
        }
    }

    if(result != 0)
    {
        msgpack_free_objects(key_obj);
        msgpack_free_objects(value_obj);
    }

    return result;
}

/*************************************************************************************************/
static int deserialize_array(msgpack_context_t *context, msgpack_object_array_t *array)
{
    int result = 0;
    msgpack_object_t *value_obj;

    for(int i = 0; i < array->count; ++i)
    {
        value_obj = NULL;

        if(CHECK_FAILURE(result, deserialize_next_object(context, &value_obj)))
        {
            break;
        }

        msgpack_object_t *tmp = value_obj;
        array->entries[i] = value_obj;
        value_obj = NULL;

        if(MSGPACK_IS_DICT(tmp) &&
           CHECK_FAILURE(result, deserialize_dict(context, (msgpack_object_dict_t*)tmp)))
        {
            break;
        }
        else if(MSGPACK_IS_ARRAY(tmp) &&
                CHECK_FAILURE(result, deserialize_array(context, (msgpack_object_array_t*)tmp)))
        {
            break;
        }
    }

    if(result != 0)
    {
        msgpack_free_objects(value_obj);
    }

    return result;
}

/*************************************************************************************************/
static int deserialize_next_object(msgpack_context_t *context, msgpack_object_t **obj_ptr)
{
    uint8_t temp_obj_buffer[sizeof(msgpack_object_bin_t)];
    uint16_t obj_size;
    msgpack_marker_t type_marker;
    bool is_dict_or_array = false;

    memset(temp_obj_buffer, 0, sizeof(temp_obj_buffer));

    RETURN_ON_FAILURE(read_bytes(context, &type_marker, sizeof(msgpack_marker_t)));

    // --------------------------
    // NIL
    // --------------------------
    if (type_marker == NIL_MARKER)
    {
        msgpack_object_t *obj  =(msgpack_object_t*)temp_obj_buffer;
        obj->type = MSGPACK_TYPE_NIL;
        obj_size = sizeof(msgpack_object_t);
    }
    // --------------------------
    // BOOL
    // --------------------------
    else if (type_marker == FALSE_MARKER || type_marker == TRUE_MARKER)
    {
        msgpack_object8_t *obj  =(msgpack_object8_t*)temp_obj_buffer;
        obj->obj.type = MSGPACK_TYPE_BOOL;
        obj->data.boolean = (type_marker == TRUE_MARKER);
        obj_size = sizeof(msgpack_object8_t);
    }
    // --------------------------
    // Fixed INT8/UINT8
    // --------------------------
    else if (type_marker <= 0x7F || type_marker >= NEGATIVE_FIXNUM_MARKER)
    {
        msgpack_object8_t *obj  =(msgpack_object8_t*)temp_obj_buffer;
        obj->obj.type = (type_marker <= 0x7F) ? MSGPACK_TYPE_UINT8 : MSGPACK_TYPE_INT8;
        obj->data.u = type_marker;
        obj_size = sizeof(msgpack_object8_t);
    }
    // --------------------------
    // 8bit values
    // --------------------------
    else if (type_marker == U8_MARKER ||
             type_marker == S8_MARKER)
    {
        msgpack_object8_t *obj  =(msgpack_object8_t*)temp_obj_buffer;
        RETURN_ON_FAILURE(read_and_convert_endian(context, (uint8_t*)&obj->data.u, sizeof(uint8_t)));
        obj->obj.type = (type_marker == U8_MARKER) ? MSGPACK_TYPE_UINT8 : MSGPACK_TYPE_INT8;
        obj_size = sizeof(msgpack_object8_t);
    }
    // --------------------------
    // 16bit values
    // --------------------------
    else if (type_marker == U16_MARKER ||
             type_marker == S16_MARKER)
    {
        msgpack_object16_t *obj  =(msgpack_object16_t*)temp_obj_buffer;
        RETURN_ON_FAILURE(read_and_convert_endian(context, (uint8_t*)&obj->data.u, sizeof(uint16_t)));
        obj->obj.type = (type_marker == U16_MARKER) ? MSGPACK_TYPE_UINT16 : MSGPACK_TYPE_INT16;
        obj_size = sizeof(msgpack_object16_t);
    }
    // --------------------------
    // 32bit values
    // --------------------------
    else if (type_marker == FLOAT_MARKER ||
             type_marker == U32_MARKER   ||
             type_marker == S32_MARKER)
    {
        msgpack_object32_t *obj  =(msgpack_object32_t*)temp_obj_buffer;
        RETURN_ON_FAILURE(read_and_convert_endian(context, (uint8_t*)&obj->data.u, sizeof(uint32_t)));
        obj->obj.type = (type_marker == FLOAT_MARKER) ? MSGPACK_TYPE_FLOAT :
                        (type_marker == U32_MARKER)   ? MSGPACK_TYPE_UINT32 : MSGPACK_TYPE_INT32;
        obj_size = sizeof(msgpack_object32_t);
    }
    // --------------------------
    // 64bit values
    // --------------------------
    else if (type_marker == DOUBLE_MARKER ||
             type_marker == U64_MARKER   ||
             type_marker == S64_MARKER)
    {
        msgpack_object64_t *obj  =(msgpack_object64_t*)temp_obj_buffer;
        RETURN_ON_FAILURE(read_and_convert_endian(context, (uint8_t*)&obj->data.u, sizeof(uint64_t)));
        obj->obj.type = (type_marker == DOUBLE_MARKER) ? MSGPACK_TYPE_DOUBLE :
                        (type_marker == U64_MARKER)    ? MSGPACK_TYPE_UINT64 : MSGPACK_TYPE_INT64;
        obj_size = sizeof(msgpack_object64_t);
    }
    // --------------------------
    // BIN (binary string)
    // --------------------------
    else if (type_marker == BIN8_MARKER ||
             type_marker == BIN16_MARKER ||
             type_marker == BIN32_MARKER)
    {
        const uint8_t read_len = (type_marker == BIN8_MARKER)  ? sizeof(uint8_t) :
                                 (type_marker == BIN16_MARKER) ? sizeof(uint16_t) : sizeof(uint32_t);

        msgpack_object_bin_t *obj  =(msgpack_object_bin_t*)temp_obj_buffer;
        RETURN_ON_FAILURE(read_and_convert_endian(context, (uint8_t*)&obj->length, read_len));
        obj->obj.type = MSGPACK_TYPE_BIN;
        obj->data = context->buffer.ptr;

        if(context->flags & MSGPACK_DESERIALIZE_WITH_PERSISTENT_STRINGS)
        {
            obj_size = sizeof(msgpack_object_bin_t) + obj->length;
        }
        else
        {
            obj_size = sizeof(msgpack_object_bin_t);
        }

        RETURN_ON_FAILURE(skip_bytes(context, obj->length));
    }

    // --------------------------
    // DICT
    // --------------------------
    else if (type_marker <= 0x8F ||
             type_marker == MAP16_MARKER ||
             type_marker == MAP32_MARKER)
    {
        msgpack_object_dict_t *obj  =(msgpack_object_dict_t*)temp_obj_buffer;

        if(type_marker <= 0x8F)
        {
            obj->count = type_marker & FIXMAP_SIZE;
        }
        else
        {
            const uint8_t read_len = (type_marker == MAP16_MARKER) ? sizeof(uint16_t) : sizeof(uint32_t);
            RETURN_ON_FAILURE(read_and_convert_endian(context, (uint8_t*)&obj->count, read_len));
        }

        is_dict_or_array = true;
        obj->obj.type = MSGPACK_TYPE_DICT;
        obj_size = sizeof(msgpack_object_dict_t) + sizeof(msgpack_dict_entry_t)*obj->count;
    }
    // --------------------------
    // ARRAY
    // --------------------------
    else if (type_marker <= 0x9F ||
            type_marker == ARRAY16_MARKER ||
            type_marker == ARRAY32_MARKER)
    {
        msgpack_object_array_t *obj  =(msgpack_object_array_t*)temp_obj_buffer;

        if(type_marker <= 0x9F)
        {
            obj->count = type_marker & FIXARRAY_SIZE;
        }
        else
        {
            const uint8_t read_len = (type_marker == ARRAY16_MARKER) ? sizeof(uint16_t) : sizeof(uint32_t);
            RETURN_ON_FAILURE(read_and_convert_endian(context, (uint8_t*)&obj->count, read_len));
        }

        is_dict_or_array = true;
        obj->obj.type = MSGPACK_TYPE_ARRAY;
        obj_size = sizeof(msgpack_object_array_t) + sizeof(msgpack_object_t*)*obj->count;
    }
    // --------------------------
    // STR (UTF8 string)
    // --------------------------
    else if (type_marker <= 0xBF ||
             type_marker == STR8_MARKER ||
             type_marker == STR16_MARKER ||
             type_marker == STR32_MARKER)
    {
        msgpack_object_str_t *obj  =(msgpack_object_str_t*)temp_obj_buffer;

        if(type_marker <= 0xBF)
        {
            obj->length = type_marker & FIXSTR_SIZE;
        }
        else
        {
            const uint8_t read_len = (type_marker == STR8_MARKER)  ? sizeof(uint8_t) :
                                     (type_marker == STR16_MARKER) ? sizeof(uint16_t) : sizeof(uint32_t);

            RETURN_ON_FAILURE(read_and_convert_endian(context, (uint8_t*)&obj->length, read_len));
        }
        obj->obj.type = MSGPACK_TYPE_STR;
        obj->data = (char*)context->buffer.ptr;

        if(context->flags & MSGPACK_DESERIALIZE_WITH_PERSISTENT_STRINGS)
        {
            obj_size = sizeof(msgpack_object_str_t) + obj->length + 1;
        }
        else
        {
            obj_size = sizeof(msgpack_object_str_t);
        }

        RETURN_ON_FAILURE(skip_bytes(context, obj->length));
    }
    else
    {
        return -1;
    }


    msgpack_object_t *obj;

    // If this is a dictionary or array then allocate space for user context pointer
    if(is_dict_or_array)
    {
        obj_size += sizeof(msgpack_user_context_t);
    }

    obj = malloc(obj_size);
    if(obj == NULL)
    {
        return -1;
    }
    memset(obj, 0, obj_size);

    memcpy(obj, temp_obj_buffer, MIN(obj_size, sizeof(temp_obj_buffer)));
    obj->flags |= MSGPACK_OBJECT_FLAG_WAS_ALLOCATED;

    if(context->flags & MSGPACK_DESERIALIZE_WITH_PERSISTENT_STRINGS)
    {
        if(obj->type == MSGPACK_TYPE_STR)
        {
            msgpack_object_str_t *str_obj = (msgpack_object_str_t*)obj;
            const char *source = str_obj->data;
            str_obj->data = (char*)&str_obj[1];
            memcpy(str_obj->data, source, str_obj->length);
        }
        else if(obj->type == MSGPACK_TYPE_BIN)
        {
            msgpack_object_bin_t *bin_obj = (msgpack_object_bin_t*)obj;
            const uint8_t *source = bin_obj->data;
            bin_obj->data = (char*)&bin_obj[1];
            memcpy(bin_obj->data, source, bin_obj->length);
        }
    }

    *obj_ptr = obj;

    return 0;
}

/*************************************************************************************************/
static int read_and_convert_endian(msgpack_context_t *context, uint8_t *data, uint32_t length)
{
    uint8_t buffer[sizeof(uint64_t)];
    int result;
    if(!CHECK_FAILURE(result, read_bytes(context, buffer, length)))
    {
        HTON_BUFFER(buffer, data, length);
    }

    return result;
}

/*************************************************************************************************/
static int read_bytes(msgpack_context_t *context, uint8_t *data, uint32_t length)
{
    if(length > MSGPACK_BUFFER_REMAINING(context))
    {
        return -1;
    }

    memcpy(data, context->buffer.ptr, length);
    context->buffer.ptr += length;

    return 0;
}

/*************************************************************************************************/
static int skip_bytes(msgpack_context_t *context, uint32_t length)
{
    if(length > MSGPACK_BUFFER_REMAINING(context))
    {
        return -1;
    }
    context->buffer.ptr += length;

    return 0;
}