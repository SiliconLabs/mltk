
#pragma once


#include <stdint.h>
#include <stdbool.h>



namespace jlink_stream
{



typedef void (StreamDataCallback)(const char *name, uint32_t length, void *arg);
typedef void (StreamConnectionCallback)(const char *name, bool connected, void *arg);

enum class StreamDirection : uint8_t
{
    Write,
    Read,
};

constexpr const StreamDirection Write = StreamDirection::Write;
constexpr const StreamDirection Read = StreamDirection::Read;



bool initialize(void);

bool register_stream(const char *name, StreamDirection direction,
                    StreamDataCallback* data_callback = nullptr,
                    StreamConnectionCallback* connection_callback = nullptr, 
                    void *arg = nullptr);
bool unregister_stream(const char *name);

bool is_connected(const char *name, bool *connected_ptr);
bool get_bytes_available(const char *name, uint32_t *bytes_available_ptr);
bool write(const char *name, const void *data, uint32_t length, uint32_t *bytes_written_ptr=nullptr);
bool write_all(const char *name, const void *data, uint32_t length);
bool read(const char *name, void *data, uint32_t length, uint32_t *bytes_read_ptr=nullptr);
void process_command(void);

} // namespace jlink_stream
