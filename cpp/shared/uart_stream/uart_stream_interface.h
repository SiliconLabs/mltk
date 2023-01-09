
#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>


#define uint32_t unsigned long



namespace uart_stream 
{

#pragma pack(1)

constexpr const uint8_t PACKET_DELIMITER1[2] = {0xDE, 0xAD};
constexpr const uint8_t PACKET_DELIMITER2[2] = {0xBE, 0xEF};

constexpr const int32_t REQUEST_SYNCHRONIZATION = -1;
constexpr const int32_t ACKNOWLEDGE_SYNCHRONIZATION = 0;
constexpr const unsigned COMMAND_PAYLOAD_LENGTH = 6;


struct PacketHeader
{
    uint8_t delimiter1[2];
    int8_t id;
    union
    {
        struct 
        {
            int8_t ack_id;
            int16_t packet_length;
            int32_t rx_buffer_available;
        } data;
        struct 
        {
            uint8_t code;
            uint8_t payload[COMMAND_PAYLOAD_LENGTH];
        } cmd;
    };
    uint8_t delimiter2[2];

    PacketHeader(
        uint8_t id,
        uint8_t ack_id,
        int32_t packet_length=0, 
        int32_t rx_buffer_available=0
    )
    : id(id)
    {
        assert(id > 0);
        data.ack_id = ack_id;
        data.packet_length = packet_length;
        data.rx_buffer_available = rx_buffer_available;
        delimiter1[0] = PACKET_DELIMITER1[0];
        delimiter1[1] = PACKET_DELIMITER1[1];
        delimiter2[0] = PACKET_DELIMITER2[0];
        delimiter2[1] = PACKET_DELIMITER2[1];
    }

    PacketHeader(
        int8_t id,
        uint8_t code,
        const uint8_t* payload = nullptr
    ) : id(id)
    {
        assert(id < 0);
        cmd.code = code;
        if(payload == nullptr)
        {
            memset(cmd.payload, 0, sizeof(cmd.payload));
        }
        else 
        {
            memcpy(cmd.payload, payload, sizeof(cmd.payload));
        }
        delimiter1[0] = PACKET_DELIMITER1[0];
        delimiter1[1] = PACKET_DELIMITER1[1];
        delimiter2[0] = PACKET_DELIMITER2[0];
        delimiter2[1] = PACKET_DELIMITER2[1];
    }
};

#pragma pack()

} // namespace uart_stream 