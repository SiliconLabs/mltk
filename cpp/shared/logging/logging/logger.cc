#include <cstring>
#include <cstdio>

#include "cpputils/string.hpp"
#include "logging/logger.hpp"




namespace logging 
{


extern void default_log_writer(const char *msg, int length, void *arg);


static const char *const log_levels[(unsigned)Level::Count] =
{
    "debug",
    "info",
    "warn",
    "error",
    "disabled"
};

/*************************************************************************************************/
Logger::Logger(const char *tag, Level level, Flags flags) : _level(level), _flags(flags)
{
    strncpy(_tag, tag, MAX_TAG_LENGTH);
    _tag[MAX_TAG_LENGTH] = 0;
    _writer = default_log_writer;
}

/*************************************************************************************************/
bool Logger::level(Level level)
{
    if (level >= Level::Count)
    {
        return false;
    }

    _level = level;

    return true;
}

/*************************************************************************************************/
bool Logger::level(const char *level)
{
    for (unsigned i = 0; i < (unsigned)Level::Count; ++i)
    {
        if (strcasecmp(log_levels[i], level) == 0)
        {
            _level = (Level)i;
            return true;
        }
    }

    return false;
}

/*************************************************************************************************/
bool Logger::level_is_enabled(Level level) const
{
    return level >= _level;
}

/*************************************************************************************************/
const char *Logger::level_str(void) const
{
    return log_levels[(unsigned)_level];
}

/*************************************************************************************************/
Flags& Logger::flags(const Flags& flags)
{
    _flags = flags;
    return _flags;
}

/*************************************************************************************************/
void Logger::writer(const Writer& writer, void *arg)
{
    _writer = writer;
    _writer_arg = arg;
}

/*************************************************************************************************/
const Writer& Logger::writer() const
{
    return _writer;
}

/*************************************************************************************************/
void Logger::debug(const char *fmt, ...) const
{
    va_list args;

    va_start(args, fmt);
    vwrite(Level::Debug, fmt, args);
    va_end(args);
}

/*************************************************************************************************/
void Logger::info(const char *fmt, ...) const
{
    va_list args;

    va_start(args, fmt);
    vwrite(Level::Info, fmt, args);
    va_end(args);
}

/*************************************************************************************************/
void Logger::warn(const char *fmt, ...) const
{
    va_list args;

    va_start(args, fmt);
    vwrite(Level::Warn, fmt, args);
    va_end(args);
}

/*************************************************************************************************/
void Logger::error(const char *fmt, ...) const
{
    va_list args;

    va_start(args, fmt);
    vwrite(Level::Error, fmt, args);
    va_end(args);
}

/*************************************************************************************************/
void Logger::dump_hex(const void *address, unsigned length, const char *desc, ...)
{
    va_list args;

    va_start(args, desc);
    vdump_hex(address, length, desc, args);
    va_end(args);
}

/*************************************************************************************************/
void Logger::vdump_hex(const void *address, unsigned length, const char *desc, va_list args)
{
    int i;
    uint8_t buff[17];
    const auto old_flags = flags();
    const uint8_t *pc = (const uint8_t *)address;


    // Output description if given.
    if (desc != nullptr)
    {
        _flags = Flag::Newline;;
        vwrite(Level::Debug, desc, args);
    }

    // Length checks.

    if (length == 0)
    {
        _flags = Flag::Newline;;
        write(Level::Debug, "  ZERO LENGTH");
        goto exit;
    }

    _flags = Flag::None;

    // Process every byte in the data.
    for (i = 0; i < length; i++)
    {
        // Multiple of 16 means new line (with line offset).
        if ((i % 16) == 0)
        {
            // Don't print ASCII buffer for the "zeroth" line.
            if (i != 0)
            {
                write(Level::Debug, "  %s\n", buff);
            }

            // Output the offset.
            write(Level::Debug, "  %04X ", i);
        }

        // Now the hex code for the specific character.
       write(Level::Debug, " %02X", pc[i]);

        // And buffer a printable ASCII character for later.

        if ((pc[i] < 0x20) || (pc[i] > 0x7e))
        {
            buff[i % 16] = '.';
        }
        else
        {
            buff[i % 16] = pc[i];
        }

        buff[(i % 16) + 1] = '\0';
    }

    // Pad out last line if not exactly 16 characters.
    while ((i % 16) != 0)
    {
        write(Level::Debug, "   ");
        i++;
    }

    // And print the final ASCII buffer.
    write(Level::Debug, "  %s\n", buff);

exit:
    _flags = old_flags;
}

/*************************************************************************************************/
void Logger::write(Level level, const char *fmt, ...) const
{
    va_list args;

    va_start(args, fmt);
    vwrite(level, fmt, args);
    va_end(args);
}

/*************************************************************************************************/
void Logger::vwrite(Level level, const char *fmt, va_list args) const
{
    if ((_writer != nullptr) && (_level != Level::Disabled) && (level >= _level))
    {
        char fmt_buffer[MAX_FMT_LENGTH];
        char *ptr = fmt_buffer;

        const int prefix_len = write_prefix(level, fmt_buffer, sizeof(fmt_buffer));
        ptr += prefix_len;

        const int max_length = MAX_FMT_LENGTH - prefix_len - 1;
        const int length = vsnprintf(ptr, max_length, fmt, args);

        if (length < max_length)
        {
            ptr += length;
            if (_flags.isSet(Flag::Newline))
            {
                *ptr++ = '\n';
            }

            _writer(fmt_buffer, (int)(ptr - fmt_buffer), _writer_arg);
        }
    }
}

/*************************************************************************************************/
void Logger::write_buffer(Level level, const char *buffer, int length) const
{
    if ((_writer != nullptr) && (_level != Level::Disabled) && (level >= _level))
    {
        char prefix_buffer[64];
        const int prefix_len = write_prefix(level, prefix_buffer, sizeof(prefix_buffer));

        if (prefix_len > 0)
        {
            _writer(prefix_buffer, prefix_len, _writer_arg);
        }

        if (length < 0)
        {
            length = strlen(buffer);
        }

        _writer(buffer, length, _writer_arg);

        if (_flags.isSet(Flag::Newline))
        {
            _writer("\n", 1, _writer_arg);
        }
    }
}

/*************************************************************************************************/
int Logger::write_prefix(Level level, char *buffer, unsigned buffer_length) const
{
    static const char level_str[] = {'D', 'I', 'W', 'E', '\0'};
    char *ptr = buffer;

    if (_flags.isSet(Flag::PrintLevel))
    {
        *ptr++ = level_str[(uint8_t)level];
        *ptr++ = ' ';
        buffer_length -= 2;
    }

    if (_flags.isSet(Flag::PrintTag))
    {
        ptr += snprintf(ptr, buffer_length, "[%s] ", _tag);
    }

    return (int)(ptr - buffer);
}

} // namespace logging 
