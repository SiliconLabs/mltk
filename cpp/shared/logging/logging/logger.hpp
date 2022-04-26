#pragma once 

#include "logging/logging.hpp"


namespace logging
{


class Logger
{
public:
    Logger(const char* tag, Level level = Level::Info, Flags flags = Flag::Newline);


    bool level(Level level);
    bool level(const char* level);
    bool level_is_enabled(Level level) const;
    const char* level_str(void) const;
    Flags& flags(const Flags& flags);
    void writer(const Writer& writer, void *arg = nullptr);
    const Writer& writer() const;

    
    void debug(const char *fmt, ...) const;
    void info(const char *fmt, ...) const;
    void warn(const char *fmt, ...) const;
    void error(const char *fmt, ...) const;
    void dump_hex(const void * address, unsigned length, const char *desc=nullptr, ...);
    void vdump_hex(const void * address, unsigned length, const char *desc, va_list args);

    void write(Level level, const char *fmt, ...) const;
    void vwrite(Level level, const char *fmt, va_list args) const;
    void write_buffer(Level level, const char *buffer, int length = -1) const;

    Flags& flags(void)
    {
        return _flags;
    }

    const Flags& flags(void) const
    {
        return _flags;
    }

    constexpr Level level(void) const
    {
        return _level;
    }

    constexpr const char* tag() const 
    {
        return _tag;
    }

protected:
    Level _level = Level::Info;
    Writer _writer = nullptr;
    void* _writer_arg = nullptr;
    char _tag[MAX_TAG_LENGTH + 1] = "";
    Flags _flags;


    int write_prefix(Level level, char *buffer, unsigned buffer_length) const;
};


} // namespace logging
