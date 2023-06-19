
#include "msgpack_internal.h"
#include "dynamic_buffer.h"

typedef struct
{
    msgpack_context_t msgpack; // must come first
    dynamic_buffer_t dynamic_buffer;
} buffered_writer_context_t;



static int buffered_msgpack_writer(void *user, const void *data, uint32_t length);



/*************************************************************************************************/
int msgpack_buffered_writer_init(msgpack_context_t **context_ptr, uint32_t initial_length)
{
    int result;
    buffered_writer_context_t *context;

    *context_ptr = NULL;
    context = malloc(sizeof(buffered_writer_context_t));
    if(context == NULL)
    {
        return -1;
    }

    memset(context, 0, sizeof(buffered_writer_context_t));

    if(CHECK_FAILURE(result, dynamic_buffer_alloc(&context->dynamic_buffer, initial_length)))
    {
    }
    else
    {
        context->msgpack = msgpack_init_with_writer(buffered_msgpack_writer, context);

        context->msgpack.buffer.buffer = context->dynamic_buffer.buffer;
        context->msgpack.buffer.ptr = context->dynamic_buffer.append;
        context->msgpack.buffer.end = (uint8_t*)context->dynamic_buffer.buffer_end;
        context->msgpack.flags |= _MSGPACK_BUFFERED_WRITER;

        *context_ptr = (msgpack_context_t*)context;
    }

    return result;
}

/*************************************************************************************************/
int msgpack_buffered_writer_deinit(msgpack_context_t *context, bool free_buffer)
{
    buffered_writer_context_t *buf_context = (buffered_writer_context_t*)context;

    if(buf_context != NULL)
    {
        if(!(buf_context->msgpack.flags & _MSGPACK_BUFFERED_WRITER))
        {
            return -1;
        }

        if(free_buffer)
        {
            dynamic_buffer_free(&buf_context->dynamic_buffer);
        }

        free(buf_context);
    }

    return 0;
}

/*************************************************************************************************/
int msgpack_buffered_writer_get_buffer(const msgpack_context_t *context, uint8_t** buffer_ptr, uint32_t* length_ptr)
{
    int result;
    buffered_writer_context_t *buf_context = (buffered_writer_context_t*)context;

    *buffer_ptr = NULL;
    *length_ptr = 0;

    if(buf_context == NULL || !(buf_context->msgpack.flags & _MSGPACK_BUFFERED_WRITER) || buf_context->dynamic_buffer.buffer == NULL)
    {
        result = -1;
    }
    else if(CHECK_FAILURE(result, dynamic_buffer_get_buffer(&buf_context->dynamic_buffer, buffer_ptr, length_ptr)))
    {
    }

    return result;
}



/** --------------------------------------------------------------------------------------------
 *  Internal functions
 * -------------------------------------------------------------------------------------------- **/


/*************************************************************************************************/
static int buffered_msgpack_writer(void *user, const void *data, uint32_t length)
{
    buffered_writer_context_t *context = user;

    RETURN_ON_FAILURE(dynamic_buffer_write(&context->dynamic_buffer, data, length));

    context->msgpack.buffer.buffer = context->dynamic_buffer.buffer;
    context->msgpack.buffer.ptr = context->dynamic_buffer.append;
    context->msgpack.buffer.end = (uint8_t*)context->dynamic_buffer.buffer_end;

    return 0;
}