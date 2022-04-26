
#pragma once


#include <stdint.h>



namespace jlink_stream
{

constexpr const uint32_t CONTEXT_MAGIC_NUMBER           = 0xF3f37BA2UL;
constexpr const uint32_t STREAM_BUFFER_MAGIC_NUMBER     = 0xE3D37B32UL;
constexpr const uint32_t STREAM_BUFFER_LENGTH           = (8*1024);


enum class StreamStatus: uint32_t
{
    InitRequired        = 0,
    InvokeInit          = ~CONTEXT_MAGIC_NUMBER,
    InitFailed          = 1,
    Idle                = 2,
    CommandReady        = 3,
    CommandExecuting    = 4,
    CommandComplete     = 5,
};

enum class StreamCommand: uint32_t
{
    OpenForRead         = 1,
    OpenForWrite        = 2,
    Close               = 3,
    GetBufferStatus     = 4,
    ReadBuffer          = 5,
    WriteBuffer         = 6
};


enum class StreamCommandResult: uint32_t
{
    Success             = 0,
    Error               = 1,
    UnknownCommand      = 2,
    NotFound            = 3,
    ReadOnly            = 4,
    WriteOnly           = 5,
    BadArgs             = 6,
    MallocFailed        = 7,
    AlreadyOpened       = 8,
    NotOpened           = 9,
};



struct StreamContextHeader
{
    uint32_t magic_number;
    StreamStatus status;
    uint32_t trigger_address;
    uint32_t trigger_value;
    StreamCommand command_code;
    StreamCommandResult command_result;
    uint32_t command_length;
    uint32_t command_buffer_address;
};



struct StreamBufferHeader
{
    uint32_t magic_number;
    uint32_t id;
    uint32_t start;
    uint32_t end;
    volatile uint32_t head;
    volatile uint32_t tail;
    volatile uint32_t length;
};




} // namespace jlink_stream

