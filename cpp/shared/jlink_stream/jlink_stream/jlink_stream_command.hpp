#pragma once 

#include <cstdint>


#include "logging/logger.hpp"
#include "jlink_stream/jlink_stream_interface.hpp"



namespace jlink_stream
{


class JlinkStreamCommand
{
public:
    static constexpr const unsigned MAX_LENGTH = STREAM_BUFFER_LENGTH;

    JlinkStreamCommand(const char* cmd_stream, const char* res_stream, logging::Logger* logger = nullptr);

    bool enable(void);
    void disable(void);

    bool receive_command(uint8_t* &data, uint32_t &length);
    bool send_response(const void* data, uint32_t length);

    bool is_connected(void) const
    {
        return connected;
    }

protected:
    const char* cmd_stream;
    const char* res_stream;
    uint32_t read_size;
    bool connected;
    logging::Logger* logger;

    static void connection_callback(const char *name, bool connected, void *arg);

};


} // namespace jlink_stream