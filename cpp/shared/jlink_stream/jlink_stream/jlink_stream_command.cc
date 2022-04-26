#include <cstdlib>

#include "jlink_stream_command.hpp"
#include "jlink_stream.hpp"




#define LOG_ERROR(msg, ...) if(logger != nullptr) logger->error(msg, ## __VA_ARGS__)
#define LOG_INFO(msg, ...) if(logger != nullptr) logger->info(msg, ## __VA_ARGS__)
#define LOG_DEBUG(msg, ...) if(logger != nullptr) logger->debug(msg, ## __VA_ARGS__)


namespace jlink_stream
{



/*************************************************************************************************/
JlinkStreamCommand::JlinkStreamCommand(const char* cmd_stream, const char* res_stream, logging::Logger* logger) :
    cmd_stream(cmd_stream), res_stream(res_stream), read_size(0), connected(false), logger(logger)
{
}

/*************************************************************************************************/
bool JlinkStreamCommand::enable(void)
{
    jlink_stream::initialize();

    if(!jlink_stream::register_stream(
        cmd_stream, StreamDirection::Read, 
        nullptr, connection_callback, 
        (void*)&connected)
    )
    {
        LOG_ERROR("Failed to register stream: %s", cmd_stream);
        return false;
    }
    if(!jlink_stream::register_stream(
        res_stream, StreamDirection::Write, 
        nullptr, nullptr, nullptr))
    {
        LOG_ERROR("Failed to register stream: %s", res_stream);
        return false;
    }

    return true;
}

/*************************************************************************************************/
void JlinkStreamCommand::disable(void)
{
    jlink_stream::unregister_stream(cmd_stream);
    jlink_stream::unregister_stream(res_stream);
}


/*************************************************************************************************/
bool JlinkStreamCommand::receive_command(uint8_t* &data, uint32_t &length)
{
    uint32_t bytes_available = 0;


    data = nullptr;
    length = 0;


    if(!jlink_stream::get_bytes_available(cmd_stream, &bytes_available))
    {
        return false;
    }

    if(read_size == 0)
    {
        volatile uint32_t bytes_read;

        if(bytes_available < sizeof(uint32_t))
        {
            return false;
        }

        if(!jlink_stream::read(cmd_stream, &read_size, sizeof(uint32_t), (uint32_t*)&bytes_read))
        {
            LOG_ERROR("Failed to read cmd length bytes");
        }


        if(bytes_read != sizeof(uint32_t) || read_size == 0 || read_size > STREAM_BUFFER_LENGTH)
        {
            read_size = 0;
            return false;
        }

        LOG_DEBUG("Command length: %d", read_size);
        if(!jlink_stream::get_bytes_available(cmd_stream, &bytes_available))
        {
            LOG_ERROR("Failed to read available");
        }
    }

    if(bytes_available < read_size)
    {
        return false;
    }


    length = read_size;
    data = static_cast<uint8_t*>(malloc(read_size));
    if(data == nullptr)
    {
        LOG_ERROR("Failed to alloc cmd buffer of size: %d", read_size);
        read_size = 0;
        return true;
    }

    volatile uint32_t bytes_read = 0;
    if(!jlink_stream::read(cmd_stream, data, read_size, (uint32_t*)&bytes_read))
    {
        LOG_ERROR("Failed to read command data");
        free(data);
        data = nullptr;
    }
    else if(read_size != bytes_read)
    {
        LOG_ERROR("Command data read bad length: %d != %d", read_size, bytes_read);
        free(data);
        data = nullptr;
    }

    read_size = 0;

    return true;
}

/*************************************************************************************************/
bool JlinkStreamCommand::send_response(const void* data, uint32_t length)
{
    auto ptr = static_cast<const uint8_t*>(data);


    if(!jlink_stream::write(res_stream, &length, sizeof(uint32_t), nullptr))
    {
        return false;
    }


    while(length > 0)
    {
        uint32_t bytes_written;

        if(!jlink_stream::write(res_stream, ptr, length, &bytes_written))
        {
            return false;
        }
        else
        {
            ptr += bytes_written;
            length -= bytes_written;
        }
    }

    return true;
}

/*************************************************************************************************/
void JlinkStreamCommand::connection_callback(const char *name, bool connected, void *arg)
{
    bool *connected_ptr = (bool*)arg;

    *connected_ptr = connected;
}


} // namespace jlink_stream
