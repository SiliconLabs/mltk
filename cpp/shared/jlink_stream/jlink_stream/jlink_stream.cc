
#include <cstdint>
#include <stdlib.h>
#include <string.h>


#include "cpputils/helpers.hpp"
#include "jlink_stream.hpp"
#include "jlink_stream_interface.hpp"
#include "jlink_stream_internal.hpp"


#define COMMAND_BUFFER_LENGTH 96


namespace jlink_stream
{


struct StreamBuffer
{
    struct StreamBuffer *next;
    const char *name;
    StreamBufferHeader *header;
    StreamDataCallback* data_callback;
    StreamConnectionCallback* connection_callback;
    void *arg;
    uint8_t id;
    StreamDirection direction;
    bool opened;
};


struct StreamContext
{
    volatile StreamContextHeader header;
    StreamBuffer *streams;
    StreamBuffer *last_used_stream;
    volatile uint32_t buffer_status_mask;
    uint32_t available_ids;
    uint8_t command_buffer[COMMAND_BUFFER_LENGTH];
};




static bool    find_stream_by_name(const char *name, StreamBuffer **stream_ptr);
static uint8_t get_next_available_id(void);



static StreamContext context;




/*************************************************************************************************/
bool initialize(void)
{
    if(context.header.magic_number == CONTEXT_MAGIC_NUMBER)
    {
        return true;
    }

    jlink_stream_internal_init((uint32_t*)&context.header.trigger_address,
                               (uint32_t*)&context.header.trigger_value,
                               (uint32_t)&context.header);

    context.available_ids = 0;
    context.header.command_buffer_address = (uint32_t)context.command_buffer;
    context.header.magic_number           = CONTEXT_MAGIC_NUMBER;

    return true;
}

/*************************************************************************************************/
bool register_stream(const char *name, StreamDirection direction,
                          StreamDataCallback* data_callback,
                          StreamConnectionCallback* connection_callback, void *arg)
{
    StreamBuffer *stream;
    StreamBufferHeader *header;

    initialize();

    // Check if the stream is already registered
    if(find_stream_by_name(name, &stream))
    {
        return false;
    }

    const uint32_t alloc_size = ALIGN_4(sizeof(StreamBuffer)) +
                                ALIGN_4(sizeof(StreamBufferHeader)) +
                                STREAM_BUFFER_LENGTH +
                                strlen(name) + 1;
    uint8_t *ptr = (uint8_t*)malloc(alloc_size);
    if(ptr == nullptr)
    {
        return false;
    }

    stream = (StreamBuffer*)ptr;
    ptr += ALIGN_4(sizeof(StreamBuffer));

    header = (StreamBufferHeader*)ptr;
    ptr += ALIGN_4(sizeof(StreamBufferHeader));

    const uint32_t buffer_start_address = (uint32_t)ptr;
    ptr += STREAM_BUFFER_LENGTH;

    stream->name = (char*)ptr;

    header->start = buffer_start_address;
    header->end  =  buffer_start_address  + STREAM_BUFFER_LENGTH;
    header->length = 0;
    header->id = get_next_available_id();
    header->magic_number = 0;

    stream->data_callback = data_callback;
    stream->connection_callback = connection_callback;
    stream->arg         = arg;
    stream->direction   = direction;
    stream->header      = header;
    stream->id          = header->id;
    stream->opened      = false;
    strcpy((char*)stream->name, name);

    stream->next = context.streams;
    context.streams = stream;

    return true;
}

/*************************************************************************************************/
bool unregister_stream(const char *name)
{
    bool retval = false;
    StreamBuffer *prev = nullptr;

    for(StreamBuffer *stream = context.streams; stream != nullptr; prev = stream, stream = stream->next)
    {
        if(strcmp(stream->name, name) == 0)
        {
            if(prev == nullptr)
            {
                context.streams = stream->next;
            }
            else
            {
                prev->next = stream->next;
            }

            if(context.last_used_stream == stream)
            {
                context.last_used_stream = nullptr;
            }

            stream->header->magic_number = 0;

            context.available_ids &= ~(1 << stream->id);
            free(stream);

            retval = true;
            break;
        }
    }

    return retval;
}

/*************************************************************************************************/
bool is_connected(const char *name, bool *connected_ptr)
{
    StreamBuffer *stream;

    *connected_ptr = false;

    if(!find_stream_by_name(name, &stream))
    {
        return false;
    }
    else
    {
        *connected_ptr = stream->opened;
        return true;
    }
}

/*************************************************************************************************/
bool get_bytes_available(const char *name, uint32_t *bytes_available_ptr)
{
    StreamBuffer *stream;

    *bytes_available_ptr = 0;

    if(!find_stream_by_name(name, &stream))
    {
        return false;
    }
    else if(!stream->opened)
    {
        return false;
    }

    const StreamBufferHeader *header = stream->header;

    jlink_stream_set_interrupt_enabled(false);
    if(stream->direction == StreamDirection::Write)
    {
        *bytes_available_ptr =STREAM_BUFFER_LENGTH - header->length;
    }
    else
    {
        *bytes_available_ptr = header->length;
    }
    jlink_stream_set_interrupt_enabled(true);

    return true;
}

/*************************************************************************************************/
bool write(const char *name, const void *data, uint32_t length, uint32_t *bytes_written_ptr)
{
    StreamBuffer *stream;
    uint32_t bytes_written_buffer;

    bytes_written_ptr = (bytes_written_ptr == nullptr) ? &bytes_written_buffer : bytes_written_ptr;

    *bytes_written_ptr = 0;


    if(!find_stream_by_name(name, &stream))
    {
        return false;
    }
    else if(!stream->opened)
    {
        return false;
    }
    else if(stream->direction != StreamDirection::Write)
    {
        return false;
    }

    jlink_stream_set_interrupt_enabled(false);

    StreamBufferHeader *header = stream->header;

    const volatile uint32_t header_length = header->length;
    const uint32_t available_space =  STREAM_BUFFER_LENGTH - header_length;
    const uint32_t write_length = MIN(available_space, length);


    if(write_length > 0)
    {
        const uint32_t length_to_end = header->end - header->tail;
        const uint8_t *data_ptr = (const uint8_t*)data;
        uint32_t remaining_length = write_length;
        const uint32_t write_chunk = MIN(length_to_end, remaining_length);

        memcpy((void*)header->tail, data_ptr, write_chunk);
        remaining_length -= write_chunk;

        header->tail += write_chunk;
        if(header->tail >= header->end)
        {
            header->tail = header->start;
        }

        if(remaining_length > 0)
        {
            data_ptr += write_chunk;
            memcpy((void*)header->tail, data_ptr, remaining_length);
            header->tail += remaining_length;
        }

        *bytes_written_ptr = write_length;
        header->length += write_length;

        context.buffer_status_mask |= (1 << stream->id);
    }

    jlink_stream_set_interrupt_enabled(true);

    return true;
}

/*************************************************************************************************/
bool write_all(const char *name, const void *data, uint32_t length)
{
    StreamBuffer *stream;

    if(!find_stream_by_name(name, &stream))
    {
        return false;
    }
    else if(stream->direction != StreamDirection::Write)
    {
        return false;
    }

    uint32_t total_remaining_length = length;
    const uint8_t *data_ptr = (const uint8_t*)data;

    while(total_remaining_length > 0)
    {
        if(!stream->opened)
        {
            return false;
        }

    
        jlink_stream_set_interrupt_enabled(false);

        StreamBufferHeader *header = stream->header;
        const volatile uint32_t header_length = header->length;
        const uint32_t available_space = STREAM_BUFFER_LENGTH - header_length;
        const uint32_t write_chunk_length = MIN(available_space, total_remaining_length);
        total_remaining_length -= write_chunk_length;

        if(write_chunk_length > 0)
        {
            const uint32_t length_to_end = header->end - header->tail;
            const uint32_t chunk_length_to_end = MIN(length_to_end, write_chunk_length);
            const uint32_t chunk_remaining_length = write_chunk_length - chunk_length_to_end;

            memcpy((void*)header->tail, data_ptr, chunk_length_to_end);
            data_ptr += chunk_length_to_end;
            
            header->tail += chunk_length_to_end;
            if(header->tail >= header->end)
            {
                header->tail = header->start;
            }

            if(chunk_remaining_length > 0)
            {
                memcpy((void*)header->tail, data_ptr, chunk_remaining_length);
                data_ptr += chunk_remaining_length;
                header->tail += chunk_remaining_length;
            }

    
            header->length += write_chunk_length;
            context.buffer_status_mask |= (1 << stream->id);
        }

        jlink_stream_set_interrupt_enabled(true);
    }

    return true;
}

/*************************************************************************************************/
bool read(const char *name, void *data, uint32_t length, uint32_t *bytes_read_ptr)
{
    StreamBuffer *stream;
    uint32_t bytes_read_buffer;

    bytes_read_ptr = (bytes_read_ptr == nullptr) ? &bytes_read_buffer : bytes_read_ptr;

    *bytes_read_ptr = 0;

    if(!find_stream_by_name(name, &stream))
    {
        return false;
    }
    else if(!stream->opened)
    {
        return false;
    }
    else if(stream->direction != StreamDirection::Read)
    {
        return false;
    }

    
    StreamBufferHeader *header = stream->header;

    jlink_stream_set_interrupt_enabled(false);

    const volatile uint32_t header_length = header->length;
    const uint32_t read_length = MIN(header_length, length);


    if(read_length > 0)
    {
        const uint32_t length_to_end = header->end - header->head;
        uint8_t *data_ptr = (uint8_t*)data;


        uint32_t remaining_length = read_length;
        const uint32_t read_chunk = MIN(length_to_end, remaining_length);
        memcpy(data_ptr, (void*)header->head, read_chunk);
        remaining_length -= read_chunk;

        header->head += read_chunk;
        if(header->head >= header->end)
        {
            header->head = header->start;
        }

        if(remaining_length > 0)
        {
            data_ptr += read_chunk;
            memcpy(data_ptr, (void*)header->head, remaining_length);
            header->head += remaining_length;
        }

        *bytes_read_ptr = read_length;

        header->length -= read_length;
        context.buffer_status_mask |= (1 << stream->id);
    }

    jlink_stream_set_interrupt_enabled(true);

    return true;
}


/** --------------------------------------------------------------------------------------------
 *  Internal functions
 * -------------------------------------------------------------------------------------------- **/

static StreamCommandResult process_open_command(StreamDirection direction);
static StreamCommandResult process_close_command(void);
static StreamCommandResult process_read_buffer_status_mask_command(void);
static StreamCommandResult process_read_buffer_command(void);
static StreamCommandResult process_write_buffer_command(void);
static StreamCommandResult find_stream_for_command_name(StreamBuffer **stream_ptr);
static StreamCommandResult find_stream_for_command_id(uint8_t id, StreamBuffer **stream_ptr);


/*************************************************************************************************
 * This is called in the platform's interrupt handler.
 * It is triggered by the associated Python script 
 */
void process_command(void)
{
    if(context.header.status == StreamStatus::InvokeInit)
    {
        context.header.status = StreamStatus::Idle;
        return;
    }

    if(context.header.status == StreamStatus::CommandReady)
    {
        StreamCommandResult result;
        const StreamCommand cmd = context.header.command_code;

        context.header.status = StreamStatus::CommandExecuting;

        switch(cmd)
        {
        case StreamCommand::OpenForRead:
        case StreamCommand::OpenForWrite:
            result = process_open_command(
                cmd == StreamCommand::OpenForRead ? StreamDirection::Read : StreamDirection::Write);
            break;

        case StreamCommand::Close:
            result = process_close_command();
            break;

        case StreamCommand::GetBufferStatus:
            result = process_read_buffer_status_mask_command();
            break;

        case StreamCommand::ReadBuffer:
            result = process_read_buffer_command();
            break;

        case StreamCommand::WriteBuffer:
            result = process_write_buffer_command();
            break;

        default:
            context.header.command_length = 0;
            result = StreamCommandResult::UnknownCommand;
            break;
        }

        context.header.command_result = result;
        context.header.status = StreamStatus::CommandComplete;
    }

}

/*************************************************************************************************/
static StreamCommandResult process_open_command(StreamDirection direction)
{
    StreamBuffer *stream;
    StreamBufferHeader *header;
    StreamCommandResult result = find_stream_for_command_name(&stream);

    context.header.command_length = 0;

    if(result != StreamCommandResult::Success)
    {
        return result;
    }
    // If the device opened the stream for reading, then the remote side should open for writing
    else if((stream->direction == StreamDirection::Read) && (direction != StreamDirection::Write))
    {
        return StreamCommandResult::ReadOnly;
    }
    // If the device opened the stream for writing, then the remote side should open for reading
    else if((stream->direction == StreamDirection::Write) && (direction != StreamDirection::Read))
    {
        return StreamCommandResult::WriteOnly;
    }
    else if(stream->opened)
    {
        return StreamCommandResult::AlreadyOpened;
    }

    stream->opened = true;
    header = stream->header;
    header->tail = header->head = header->start;
    header->magic_number = STREAM_BUFFER_MAGIC_NUMBER;

    const uint32_t stream_id = stream->id;
    const uint32_t base_address = (uint32_t)header;
    memcpy(&context.command_buffer[0], &stream_id, sizeof(uint32_t));
    memcpy(&context.command_buffer[4], &base_address, sizeof(uint32_t));
    context.header.command_length = sizeof(uint32_t)*2;

    // Signal that this stream is available for processing
    context.buffer_status_mask |= (1 << stream->id);

    if(stream->connection_callback != nullptr)
    {
        stream->connection_callback(stream->name, true, stream->arg);
    }

    if(stream->direction == StreamDirection::Write && stream->data_callback != nullptr)
    {
        stream->data_callback(stream->name, STREAM_BUFFER_LENGTH, stream->arg);
    }

    return StreamCommandResult::Success;
}

/*************************************************************************************************/
static StreamCommandResult process_close_command(void)
{
    StreamBuffer *stream;
    StreamCommandResult result = find_stream_for_command_name(&stream);

    if(result != StreamCommandResult::Success)
    {
        return result;
    }
    else if(!stream->opened)
    {
        return StreamCommandResult::NotOpened;
    }
    else
    {
        if(stream->connection_callback != nullptr)
        {
            stream->connection_callback(stream->name, false, stream->arg);
        }

        // Signal that this stream is available for processing
        context.buffer_status_mask &= ~(1 << stream->id);

        stream->opened = false;
        stream->header->magic_number = 0;
    }

    return StreamCommandResult::Success;
}

/*************************************************************************************************/
static StreamCommandResult process_read_buffer_status_mask_command(void)
{
    memcpy(&context.command_buffer[0], (void*)&context.buffer_status_mask, sizeof(uint32_t));
    context.header.command_length = sizeof(uint32_t);

    // Clear the buffer status after it has been read
    context.buffer_status_mask = 0;

    return StreamCommandResult::Success;
}

/*************************************************************************************************/
static StreamCommandResult process_read_buffer_command(void)
{
    uint32_t read_length;
    uint32_t stream_id;
    StreamBuffer *stream;

    memcpy(&stream_id, &context.command_buffer[0], sizeof(uint32_t));
    memcpy(&read_length, &context.command_buffer[4], sizeof(uint32_t));

    StreamCommandResult result = find_stream_for_command_id(stream_id, &stream);

    if(result != StreamCommandResult::Success)
    {
        return result;
    }
    // If the device opened the stream for reading,
    // then the remote side can only write it
    else if(stream->direction != StreamDirection::Write)
    {
         return StreamCommandResult::WriteOnly;
    }
    else if(!stream->opened)
    {
        return StreamCommandResult::NotOpened;
    }
    else
    {
        StreamBufferHeader *header = stream->header;
        uint32_t length_to_end = header->end - header->head;

        if(read_length > header->length)
        {
            return StreamCommandResult::BadArgs;
        }


        if(read_length < length_to_end)
        {
            header->head += read_length;
        }
        else
        {
            header->head = header->start + (read_length - length_to_end);
        }

        header->length -= read_length;

        if(stream->data_callback != nullptr)
        {
            stream->data_callback(stream->name, STREAM_BUFFER_LENGTH - header->length, stream->arg);
        }
    }

    return StreamCommandResult::Success;
}

/*************************************************************************************************/
static StreamCommandResult process_write_buffer_command(void)
{
    uint32_t write_length;
    uint32_t stream_id;
    StreamBuffer *stream;

    context.header.command_length = 0;
    memcpy(&stream_id, &context.command_buffer[0], sizeof(uint32_t));
    memcpy(&write_length, &context.command_buffer[4], sizeof(uint32_t));

    StreamCommandResult result = find_stream_for_command_id(stream_id, &stream);

    if(result != StreamCommandResult::Success)
    {
        return result;
    }
    // If the device opened the stream for writing,
    // then the remote side can only read it
    else if(stream->direction != StreamDirection::Read)
    {
        return StreamCommandResult::ReadOnly;
    }
    else if(!stream->opened)
    {
        return StreamCommandResult::NotOpened;
    }
    else
    {
        StreamBufferHeader *header = stream->header;
        uint32_t length_to_end = header->end - header->tail;

        if(write_length + header->length > STREAM_BUFFER_LENGTH)
        {
            return StreamCommandResult::BadArgs;
        }

        if(write_length < length_to_end)
        {
            header->tail += write_length;
        }
        else
        {
            header->tail = header->start + (write_length - length_to_end);
        }

        header->length += write_length;

        if(stream->data_callback != nullptr)
        {
            stream->data_callback(stream->name, header->length, stream->arg);
        }
    }

    return StreamCommandResult::Success;
}

/*************************************************************************************************/
static StreamCommandResult find_stream_for_command_name(StreamBuffer **stream_ptr)
{
    if(context.header.command_length == 0 || context.header.command_length >= COMMAND_BUFFER_LENGTH)
    {
        return StreamCommandResult::BadArgs;
    }

    context.command_buffer[context.header.command_length] = 0;
    context.header.command_length = 0;

    if(find_stream_by_name((char*)context.command_buffer, stream_ptr))
    {
        return StreamCommandResult::Success;
    }
    else
    {
        return StreamCommandResult::NotFound;
    }
}

/*************************************************************************************************/
static StreamCommandResult find_stream_for_command_id(uint8_t id, StreamBuffer **stream_ptr)
{
    StreamCommandResult result = StreamCommandResult::NotFound;

    for(StreamBuffer *stream = context.streams; stream != nullptr; stream = stream->next)
    {
        if(stream->id == id)
        {
            *stream_ptr = stream;
            result = StreamCommandResult::Success;
            break;
        }
    }

    return result;
}

/*************************************************************************************************/
static bool find_stream_by_name(const char *name, StreamBuffer **stream_ptr)
{
    bool retval = false;

    if(context.last_used_stream != nullptr)
    {
        if(strcmp(context.last_used_stream->name, name) == 0)
        {
            *stream_ptr = context.last_used_stream;
            retval = true;
            goto exit;
        }
    }

    for(StreamBuffer *stream = context.streams; stream != nullptr; stream = stream->next)
    {
        if((stream->name == name) || (strcmp(name, stream->name) == 0))
        {
            context.last_used_stream = stream;
            *stream_ptr = stream;
            retval = true;
            break;
        }
    }

    exit:
    return retval;
}

/*************************************************************************************************/
static uint8_t get_next_available_id(void)
{
    uint8_t retval;

    for(retval = 0; retval < 32; ++retval)
    {
        const uint32_t id_mask = (1 << retval);

        if((id_mask & context.available_ids) == 0)
        {
            context.available_ids |= id_mask;
            break;
        }
    }

    return retval;
}


} // namespace jlink_stream
