
#include "str_util.h"
#include "msgpack_internal.h"





static msgpack_user_context_t* get_user_context(const msgpack_object_t *obj);





/*************************************************************************************************/
int msgpack_set_user_context(msgpack_object_t *obj, void *context, bool auto_free)
{
    msgpack_user_context_t *user = get_user_context(obj);

    if(user == NULL)
    {
        return -1;
    }
    else
    {
        user->context = context;

        if(auto_free)
        {
            obj->flags |= MSGPACK_OBJECT_FLAG_AUTO_FREE_USER;
        }
        else
        {
            obj->flags &= ~MSGPACK_OBJECT_FLAG_AUTO_FREE_USER;
        }

        return 0;
    }
}

/*************************************************************************************************/
int msgpack_get_user_context(const msgpack_object_t *obj, void **context_ptr)
{
    const msgpack_user_context_t *user = get_user_context(obj);

    if(user == NULL)
    {
        return -1;
    }
    else
    {
        *context_ptr = (void*)user->context;

        return 0;
    }
}

/*************************************************************************************************/
int msgpack_remove_dict_object(msgpack_object_dict_t *dict_obj, void *obj)
{
    int result = -1;

    if(!MSGPACK_IS_DICT(dict_obj))
    {
        goto exit;
    }

    // Find the object in the dictionary
    for(int i = 0; i < dict_obj->count; ++i)
    {
        msgpack_dict_entry_t *entry= &dict_obj->entries[i];

        if(entry->value != obj)
        {
            continue;
        }

        // Object found, clean up the key
        if((entry->key != NULL) && (entry->key->flags & MSGPACK_OBJECT_FLAG_WAS_ALLOCATED))
        {
            free((void*)entry->key);
        }

        // Remove the pointers to the object
        entry->key = NULL;
        entry->value = NULL;

        result = 0;
        break;
    }

    exit:
    return result;
}

/*************************************************************************************************/
void msgpack_free_objects(void *obj)
{
    msgpack_object_t *object = obj;

    if(object != NULL)
    {
        bool is_dict_or_array = false;

        if(MSGPACK_IS_DICT(object))
        {
            msgpack_object_dict_t *dict = (msgpack_object_dict_t*)object;
            is_dict_or_array = true;

            for(int i = 0; i < dict->count; ++i)
            {
                const msgpack_dict_entry_t *entry= &dict->entries[i];
                if((entry->key != NULL) && (entry->key->flags & MSGPACK_OBJECT_FLAG_WAS_ALLOCATED))
                {
                    free((void*)entry->key);
                }
                msgpack_free_objects(entry->value);
            }

            if(dict->obj.flags & MSGPACK_OBJECT_FLAG_WAS_ALLOCATED)
            {
                free(dict->entries);
            }
        }
        else if(MSGPACK_IS_ARRAY(object))
        {
            msgpack_object_array_t *array = (msgpack_object_array_t*)object;
            is_dict_or_array = true;

            for(int i = 0; i < array->count; ++i)
            {
                msgpack_object_t *entry = array->entries[i];
                msgpack_free_objects(entry);
            }

            if(array->obj.flags & MSGPACK_OBJECT_FLAG_WAS_ALLOCATED)
            {
                free(array->entries);
            }
        }

        // If this object is a dict or array AND the 'AUTO_FREE_USER' flag is set
        if(is_dict_or_array && (object->flags & MSGPACK_OBJECT_FLAG_AUTO_FREE_USER))
        {
            // If a user context was specified
            msgpack_user_context_t *user = get_user_context(object);
            if(user->context != NULL)
            {
                // Then free the user context
                free(user->context);
            }
            user->context = NULL;
        }

        if(object->flags & MSGPACK_OBJECT_FLAG_WAS_ALLOCATED)
        {
            free(object);
        }
    }
}


/*************************************************************************************************/
 msgpack_object_t* msgpack_get_dict_object(const msgpack_object_dict_t *dict, const char* key, msgpack_type_t type)
{
    msgpack_object_t *retval = NULL;

    if(dict == NULL || !MSGPACK_IS_DICT(dict))
    {
        return NULL;
    }

    for(int i = 0; i < dict->count; ++i)
    {
        const msgpack_dict_entry_t *entry= &dict->entries[i];
        if(MSGPACK_STR_CMP(entry->key, key) == 0)
        {
            if(msgpack_object_is_type(entry->value, type))
            {
                retval = entry->value;
            }
            break;
        }
    }

    return retval;
}

/*************************************************************************************************/
msgpack_object_t* msgpack_get_array_object(const msgpack_object_array_t *array, uint32_t index, msgpack_type_t type)
{
    if(array == NULL || !MSGPACK_IS_ARRAY(array))
    {
        return NULL;
    }
    else if(array->count <= index)
    {
        return NULL;
    }
    else
    {
        const msgpack_object_t *entry= array->entries[index];
        return msgpack_object_is_type(entry, type) ? (msgpack_object_t*)entry : NULL;
    }
}

/*************************************************************************************************/
int msgpack_foreach(const msgpack_object_t *dict_or_array, msgpack_iterator_t iterator_callback, void *arg, uint32_t recursive_depth)
{
    int result = 0;

    if(dict_or_array == NULL)
    {
        result = -1;
    }
    else if(MSGPACK_IS_DICT(dict_or_array))
    {
        const msgpack_object_dict_t *dict = (msgpack_object_dict_t*)dict_or_array;

        for(int i = 0; i < dict->count; ++i)
        {
            const msgpack_dict_entry_t *entry= &dict->entries[i];

            if(CHECK_FAILURE(result, iterator_callback(entry->key, entry->value, arg)))
            {
                break;
            }

            if((recursive_depth > 0) &&
               (MSGPACK_IS_DICT(entry->value) || MSGPACK_IS_ARRAY(entry->value)))
            {
                if(CHECK_FAILURE(result, msgpack_foreach(entry->value, iterator_callback, arg, recursive_depth-1)))
                {
                    break;
                }
            }
        }
    }
    else if(MSGPACK_IS_ARRAY(dict_or_array))
    {
        const msgpack_object_array_t *array_obj = (const msgpack_object_array_t*)dict_or_array;

        for(uint32_t i = 0; i < MSGPACK_ARRAY_LENGTH(array_obj); ++i)
        {
            const msgpack_object_t *array_entry = array_obj->entries[i];

            if(CHECK_FAILURE(result, iterator_callback(NULL, array_entry, arg)))
            {
                break;
            }

            if((recursive_depth > 0) &&
               (MSGPACK_IS_DICT(array_entry) || MSGPACK_IS_ARRAY(array_entry)))
            {
                if(CHECK_FAILURE(result, msgpack_foreach(array_entry, iterator_callback, arg, recursive_depth-1)))
                {
                    break;
                }
            }
        }
    }
    else
    {
        result = -1;
    }

    return result;
}

/*************************************************************************************************/
char* msgpack_to_str(const msgpack_object_t *obj, char *buffer, uint32_t max_length)
{
    if(max_length < 32)
    {
        return NULL;
    }

    if(MSGPACK_IS_DICT(obj))
    {
        const msgpack_object_dict_t *dict = (msgpack_object_dict_t*)obj;
        snprintf(buffer, max_length, "<dict: %u entries>", (unsigned int)dict->count);
    }
    else if(MSGPACK_IS_ARRAY(obj))
    {
        const msgpack_object_array_t *array_obj = (const msgpack_object_array_t*)obj;
        snprintf(buffer, max_length, "<array: %u entries>", (unsigned int)array_obj->count);
    }
    else if(MSGPACK_IS_STR(obj))
    {
        MSGPACK_STR(obj, buffer, max_length);
    }
    else if(MSGPACK_IS_BIN(obj))
    {
        str_binary_to_hex_buffer(buffer, max_length, MSGPACK_BIN_VALUE(obj), MSGPACK_BIN_LENGTH(obj));
    }
    else if(MSGPACK_IS_BOOL(obj))
    {
       strcpy(buffer,  MSGPACK_BOOL(obj) ? "true" : "false");
    }
    else if(MSGPACK_IS_FLOAT(obj))
    {
        snprintf(buffer, max_length, "%.5f", MSGPACK_FLOAT(obj));
    }
    else if(MSGPACK_IS_DOUBLE(obj))
    {
       snprintf(buffer, max_length, "%.5f", MSGPACK_DOUBLE(obj));
    }
    else if(MSGPACK_IS_UINT64(obj))
    {
        uint64_t uint64;
        uint64_to_str(*MSGPACK_ULONG(obj, &uint64), buffer);
    }
    else if(MSGPACK_IS_INT(obj))
    {
        int32_to_str(MSGPACK_INT(obj), buffer);
    }
    else if(MSGPACK_IS_UINT(obj))
    {
        uint32_to_str(MSGPACK_UINT(obj), buffer);
    }
    else
    {
        strcpy(buffer, "<unknown>");
    }

    return buffer;
}

/*************************************************************************************************/
bool msgpack_object_is_type(const msgpack_object_t *object, msgpack_type_t type)
{
    bool retval;
    if(object == NULL)
    {
        retval = false;
    }
    else if(type == MSGPACK_TYPE_ANY)
    {
        retval = true;
    }
    else if(object->type == type)
    {
        retval = true;
    }
    else if(type == MSGPACK_TYPE_INT)
    {
        const msgpack_type_t obj_type = object->type;
        retval = (obj_type == MSGPACK_TYPE_INT8 || obj_type == MSGPACK_TYPE_INT16 ||
                  obj_type == MSGPACK_TYPE_INT32 || obj_type == MSGPACK_TYPE_INT64);
    }
    else if(type == MSGPACK_TYPE_UINT)
    {
        const msgpack_type_t obj_type = object->type;
        retval = (obj_type == MSGPACK_TYPE_UINT8 || obj_type == MSGPACK_TYPE_UINT16 ||
                  obj_type == MSGPACK_TYPE_UINT32 || obj_type == MSGPACK_TYPE_UINT64 );
    }
    else if(type == MSGPACK_TYPE_INT_OR_UINT)
    {
        const msgpack_type_t obj_type = object->type;
        retval = (obj_type == MSGPACK_TYPE_UINT8 || obj_type == MSGPACK_TYPE_UINT16 ||
                  obj_type == MSGPACK_TYPE_UINT32 || obj_type == MSGPACK_TYPE_UINT64 ||
                  obj_type == MSGPACK_TYPE_INT8 || obj_type == MSGPACK_TYPE_INT16 ||
                  obj_type == MSGPACK_TYPE_INT32 || obj_type == MSGPACK_TYPE_INT64 ||
                  obj_type == MSGPACK_TYPE_BOOL);
    }
    else
    {
        retval = false;
    }

    return retval;
}

/*************************************************************************************************/
int32_t msgpack_get_int(const msgpack_object_t *object)
{
    return (int32_t)msgpack_get_uint(object);
}

/*************************************************************************************************/
uint32_t msgpack_get_uint(const msgpack_object_t *object)
{
    uint32_t retval = 0;
    const msgpack_type_t type = object->type;
    void *value_ptr = (void*)&object[1];

    switch(type)
    {
    case MSGPACK_TYPE_INT8:
    {
        const int8_t *ptr = (int8_t*)value_ptr;
        retval = (uint32_t)(*ptr);
    } break;
    case MSGPACK_TYPE_BOOL:
    case MSGPACK_TYPE_UINT8:
    {
        const uint8_t *ptr = (uint8_t*)value_ptr;
        retval = (uint32_t)(*ptr);
    } break;
    case MSGPACK_TYPE_INT16:
    {
        const int16_t *ptr = (int16_t*)value_ptr;
        retval = (uint32_t)(*ptr);
    } break;
    case MSGPACK_TYPE_UINT16:
    {
        const uint16_t *ptr = (uint16_t*)value_ptr;
        retval = (uint32_t)(*ptr);
    } break;
    case MSGPACK_TYPE_INT32:
    {
        const int32_t *ptr = (int32_t*)value_ptr;
        retval = (uint32_t)(*ptr);
    } break;
    case MSGPACK_TYPE_UINT32:
    {
        const uint32_t *ptr = (uint32_t*)value_ptr;
        retval = *ptr;
    } break;
    case MSGPACK_TYPE_STR:
    {
        char str_buffer[32];
        msgpack_get_str(object, str_buffer, sizeof(str_buffer));
        retval = str_to_uint32(str_buffer);
    } break;
    default:
        break;
    }

    return retval;
}

/*************************************************************************************************/
int64_t* msgpack_get_long(const msgpack_object_t *object, int64_t *buffer)
{
    msgpack_get_ulong(object, (uint64_t*)buffer);
    return buffer;
}

/*************************************************************************************************/
uint64_t* msgpack_get_ulong(const msgpack_object_t *object, uint64_t *buffer)
{
    const msgpack_type_t type = object->type;

    if(type == MSGPACK_TYPE_STR)
    {
        char str_buffer[32];
        msgpack_get_str(object, str_buffer, sizeof(str_buffer));
        *buffer = str_to_uint64(str_buffer);
    }
    else if(type == MSGPACK_TYPE_INT64 || type == MSGPACK_TYPE_UINT64 || type == MSGPACK_TYPE_DOUBLE)
    {
        memcpy(buffer, (void*)&object[1], sizeof(uint64_t));
    }
    else
    {
        const uint32_t val = msgpack_get_uint(object);
        // if the 32bit value is negative then sign-extend into the 64bit value
        const uint8_t sign_extend = ((val & 0x80000000UL) &&
                                    ((type == MSGPACK_TYPE_INT8)  ||
                                     (type == MSGPACK_TYPE_INT16) ||
                                     (type == MSGPACK_TYPE_INT32))) ? 0xFF : 0x00;
        memset(buffer, sign_extend, sizeof(uint64_t));
        memcpy(buffer, &val, sizeof(uint32_t));
    }

    return buffer;
}

/*************************************************************************************************/
char* msgpack_get_str(const msgpack_object_t *object, char* buffer, uint16_t max_length)
{
    const int len = MIN(max_length-1, MSGPACK_STR_LENGTH(object));
    strncpy(buffer, MSGPACK_STR_VALUE(object), len);
    buffer[len] = 0;
    return buffer;
}

/*************************************************************************************************/
int msgpack_str_cmp(const msgpack_object_t* object, const char* str)
{
    if(object == NULL || object->type != MSGPACK_TYPE_STR || str == NULL)
    {
        return -2;
    }
    const uint32_t obj_str_len = MSGPACK_STR_LENGTH(object);
    const uint32_t str_len = strlen(str);
    if(obj_str_len < str_len)
    {
        return -1;
    }
    else if(obj_str_len > str_len)
    {
        return 1;
    }
    else
    {
        return strncmp(MSGPACK_STR_VALUE(object), str, str_len);
    }
}



/** --------------------------------------------------------------------------------------------
 *  Internal functions
 * -------------------------------------------------------------------------------------------- **/


/*************************************************************************************************/
static msgpack_user_context_t* get_user_context(const msgpack_object_t *obj)
{
    msgpack_user_context_t *retval = NULL;


    if(obj->type == MSGPACK_TYPE_DICT)
    {
        const msgpack_object_dict_t *dict_obj = (msgpack_object_dict_t*)obj;
        retval = (msgpack_user_context_t*)((uint8_t*)dict_obj + sizeof(msgpack_object_dict_t) + sizeof(msgpack_dict_entry_t)*dict_obj->count);
    }
    else if(obj->type == MSGPACK_TYPE_ARRAY)
    {
        const msgpack_object_array_t *array_obj = (msgpack_object_array_t*)obj;
        retval = (msgpack_user_context_t*)((uint8_t*)array_obj + sizeof(msgpack_object_array_t) + sizeof(msgpack_object_t)*array_obj->count);
    }

    return retval;
}
