
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdio.h>

#include "dynamic_buffer.h"



static int allocate_larger_buffer(dynamic_buffer_t *buffer, uint32_t additional_length);




/*************************************************************************************************/
int dynamic_buffer_alloc(dynamic_buffer_t *buffer, uint32_t length)
{
    int result;

    buffer->buffer = malloc(length);

    if(buffer->buffer == NULL)
    {
        return -1;
    }
    memset(buffer->buffer, 0, length);
    buffer->buffer_end = buffer->buffer + length;
    buffer->prepend = buffer->append = buffer->buffer;

    return 0;
}

/*************************************************************************************************/
int dynamic_buffer_realloc(dynamic_buffer_t *buffer, uint32_t additional_length)
{
    int result = 0;
    const uint32_t remaining_len = dynamic_buffer_get_remaining_length(buffer);

    if(remaining_len < additional_length)
    {
        uint8_t *new_buffer;
        const uint32_t new_length = dynamic_buffer_get_total_size(buffer) + additional_length;

        new_buffer = malloc(new_length);
        if(new_buffer == NULL)
        {
            return -1;
        }
        memset(new_buffer, 0, new_length);

        const uint32_t prepend_offset = (uint32_t)(buffer->prepend - buffer->buffer);
        const uint32_t append_offset = (uint32_t)(buffer->append - buffer->buffer);
        memcpy(new_buffer, buffer->buffer, append_offset);

        free(buffer->buffer);

        buffer->buffer = new_buffer;
        buffer->buffer_end = new_buffer + new_length;
        buffer->prepend = new_buffer + prepend_offset;
        buffer->append = new_buffer + append_offset;
    }

    return result;
}

/*************************************************************************************************/
int dynamic_buffer_copy(dynamic_buffer_t *dst, const dynamic_buffer_t *src)
{
    return dynamic_buffer_write(dst, src->prepend, dynamic_buffer_get_length(src));
}

/*************************************************************************************************/
void dynamic_buffer_free(dynamic_buffer_t *buffer)
{
    if(buffer != NULL && buffer->buffer != NULL)
    {
        free(buffer->buffer);
        buffer->buffer = NULL;
    }
}

/*************************************************************************************************/
void dynamic_buffer_reset(dynamic_buffer_t *buffer)
{
    buffer->prepend = buffer->append = buffer->buffer;
}

/*************************************************************************************************/
int dynamic_buffer_write(dynamic_buffer_t *buffer, const void *data, uint32_t data_length)
{
    if(buffer->append + data_length > buffer->buffer_end)
    {
        int result = allocate_larger_buffer(buffer, data_length);

        if(result != 0)
        {
            return result;
        }
    }

    if(data_length > 0)
    {
        memcpy(buffer->append, data, data_length);
        buffer->append += data_length;
    }

    return 0;
}

/*************************************************************************************************/
int dynamic_buffer_writef(dynamic_buffer_t *buffer, const char *fmt, ...)
{
    va_list args;

    va_start(args, fmt);
    const int result = dynamic_buffer_vwrite(buffer, fmt, args);
    va_end(args);

    return result;
}

/*************************************************************************************************/
int dynamic_buffer_vwrite(dynamic_buffer_t *buffer, const char *fmt, va_list args)
{
    int length = 0;

    for(;;)
    {
        const int remaining = (buffer->buffer_end - buffer->append);

        length = vsnprintf((char*)buffer->append, remaining-1, fmt, args);

        if(length >= remaining)
        {
            int result = allocate_larger_buffer(buffer, length);

            if(result != 0)
            {
                return result;
            }
        }
        else
        {
            break;
        }
    }

    buffer->append += length;

    return 0;
}

/*************************************************************************************************
 * Return the available data length
 */
uint32_t dynamic_buffer_get_length(const dynamic_buffer_t *buffer)
{
    return (uint32_t)(buffer->append - buffer->prepend);
}

/*************************************************************************************************
 * Return total size of allocated buffer
 */
uint32_t dynamic_buffer_get_total_size(const dynamic_buffer_t *buffer)
{
    return (uint32_t)(buffer->buffer_end - buffer->buffer);
}

/*************************************************************************************************
 * Return the remaining number of bytes that may be written to the buffer
 */
uint32_t dynamic_buffer_get_remaining_length(const dynamic_buffer_t *buffer)
{
    return (uint32_t)(buffer->buffer_end - buffer->append);
}

/*************************************************************************************************/
int dynamic_buffer_get_buffer(const dynamic_buffer_t *dynamic_buffer, uint8_t** buffer_ptr, uint32_t* length_ptr)
{
    *buffer_ptr = dynamic_buffer_get_data_start(dynamic_buffer);
    *length_ptr = dynamic_buffer_get_length(dynamic_buffer);

    return 0;
}

/*************************************************************************************************/
int dynamic_buffer_adjust_data_end(dynamic_buffer_t *buffer, int32_t adjust_amount)
{
    if((buffer->append + adjust_amount < buffer->buffer) || (buffer->append + adjust_amount > buffer->buffer_end))
    {
        return -1;
    }
    else
    {
        buffer->append += adjust_amount;
        return 0;
    }
}

/*************************************************************************************************/
int dynamic_buffer_adjust_data_start(dynamic_buffer_t *buffer, int32_t adjust_amount)
{
    if((buffer->prepend + adjust_amount < buffer->buffer) || (buffer->prepend + adjust_amount > buffer->buffer_end))
    {
        return -1;
    }
    else
    {
        buffer->prepend += adjust_amount;
        return 0;
    }
}


/** --------------------------------------------------------------------------------------------
 *  Internal functions
 * -------------------------------------------------------------------------------------------- **/

#ifndef MAX
#define MAX(x,y)  ((x) > (y) ? (x) : (y))
#endif /* ifndef MAX */
#define ALIGN_n(x, n) ((((uint32_t)x) + ((n)-1)) & ~((n)-1))


/*************************************************************************************************/
static int allocate_larger_buffer(dynamic_buffer_t *buffer, uint32_t additional_length)
{
    uint8_t *larger_buffer;
    const uint32_t prepend_offset = (uint32_t)(buffer->prepend - buffer->buffer);
    const uint32_t append_offset = (uint32_t)(buffer->append - buffer->buffer);
    uint32_t alloc_size = dynamic_buffer_get_total_size(buffer);
    const uint32_t alloc_chunk_size = MAX(ALIGN_n(additional_length, 128), alloc_size / 2);

    while(alloc_size < append_offset + additional_length)
    {
        alloc_size += alloc_chunk_size;
    }

    larger_buffer = malloc(alloc_size);
    if(larger_buffer == NULL)
    {
        return -1;
    }
    memset(&larger_buffer[append_offset], 0, alloc_size-append_offset);
    memcpy(larger_buffer, buffer->buffer, append_offset);

    free(buffer->buffer);

    buffer->buffer = larger_buffer;
    buffer->buffer_end = larger_buffer + alloc_size;
    buffer->prepend = larger_buffer + prepend_offset;
    buffer->append = larger_buffer + append_offset;

    return 0;
}