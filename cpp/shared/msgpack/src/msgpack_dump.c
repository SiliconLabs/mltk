#include <stdarg.h>
#include <stdio.h>

#include "msgpack_internal.h"


#define MAX_CONTAINER_DEPTH 32
#define MAX_ARRAY_LENGTH 32

typedef struct
{
    uint32_t max_depth;
    void (*write_callback)(const char*, void*);
    void *write_callback_arg;
} context_t;



static int dump_callback(context_t *context, const msgpack_object_t *root, int recursive_depth);
static void write_str(context_t *context, int depth, const char* fmt, ...);



/*************************************************************************************************/
int msgpack_dump(
    const msgpack_object_t *dict_or_array,
    uint32_t recursive_depth,
    void (*write_callback)(const char*, void*),
    void *write_callback_arg
)
{
    context_t context =
    {
        .max_depth = recursive_depth,
        .write_callback = write_callback,
        .write_callback_arg = write_callback_arg
    };

    if(dict_or_array == NULL || !(MSGPACK_IS_DICT(dict_or_array) || MSGPACK_IS_ARRAY(dict_or_array)))
    {
        return -1;
    }

    {
        char format_buffer[64];
        write_str(&context, 0, "%s\n", msgpack_to_str(dict_or_array, format_buffer, sizeof(format_buffer)));
    }

    return dump_callback(&context, dict_or_array, 0);
}


/*************************************************************************************************/
static int dump_callback(context_t *context, const msgpack_object_t *root, int recursive_depth)
{
    char format_buffer[64];

    if(MSGPACK_IS_DICT(root))
    {
        const msgpack_object_dict_t *dict = (msgpack_object_dict_t*)root;

        for(int i = 0; i < dict->count; ++i)
        {
            const msgpack_dict_entry_t *entry = &dict->entries[i];

            write_str(context, recursive_depth, "  %s: ", msgpack_to_str(entry->key, format_buffer, sizeof(format_buffer)));
            write_str(context, -1, "%s\n", msgpack_to_str(entry->value, format_buffer, sizeof(format_buffer)));

            if((recursive_depth < context->max_depth) &&
               (MSGPACK_IS_DICT(entry->value) || MSGPACK_IS_ARRAY(entry->value)))
            {
                RETURN_ON_FAILURE(dump_callback(context, entry->value, recursive_depth+1));
            }
        }
    }
    else if(MSGPACK_IS_ARRAY(root))
    {
        const msgpack_object_array_t *array_obj = (const msgpack_object_array_t*)root;


        for(uint32_t i = 0; i < MSGPACK_ARRAY_LENGTH(array_obj); ++i)
        {
            if(i == MAX_ARRAY_LENGTH)
            {
                write_str(context, recursive_depth, "  [%d] ...\n", i);
                break;
            }

            const msgpack_object_t *entry = array_obj->entries[i];

            write_str(context, recursive_depth, "  [%d] %s\n", i, msgpack_to_str(entry, format_buffer, sizeof(format_buffer)));

            if((recursive_depth < context->max_depth) &&
               (MSGPACK_IS_DICT(entry) || MSGPACK_IS_ARRAY(entry)))
            {
                RETURN_ON_FAILURE(dump_callback(context, entry, recursive_depth+1));
            }
        }
    }
    else
    {
        return -1;
    }

    return 0;
}

/*************************************************************************************************/
static void write_str(context_t *context, int depth, const char* fmt, ...)
{
    char format_buffer[256];
    char* ptr = format_buffer;
    va_list args;

    if(depth >= (int)(sizeof(format_buffer)/2 - 4))
    {
        return;
    }

    for(int i = 0; i < depth; ++i)
    {
        *ptr++ = ' ';
        *ptr++ = ' ';
    }

    const int max_len = (int)(&format_buffer[sizeof(format_buffer)]-ptr);

    va_start(args, fmt);
    int l = vsnprintf(ptr, max_len, fmt, args);
    va_end(args);

    if(l >= max_len)
    {
        return;
    }

    context->write_callback(format_buffer,  context->write_callback_arg);
}