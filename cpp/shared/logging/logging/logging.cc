
#include <new>
#include <cassert>
#include <cstring>
#include <cstdarg>

#include "cpputils/helpers.hpp"
#include "logging/logger.hpp"


namespace logging
{


static LoggerList* _loggers;
CREATE_STATIC_OBJECT_BUFFER(LoggerList, loggers_buffer);



/*************************************************************************************************/
void set_logger_list(LoggerList* list)
{
    _loggers = list;
}

/*************************************************************************************************/
#ifndef MLTK_DLL_IMPORT 
LoggerList& get_loggers()
{
    if(_loggers == nullptr)
    {
        _loggers = new(loggers_buffer)LoggerList();
    }
    return *_loggers;
}
#endif

/*************************************************************************************************/
Logger* create(const char* tag, Level level, Flags flags)
{
    auto& loggers = get_loggers();

    if(get(tag) != nullptr)
    {
        assert(!"Logger already exists");
        return nullptr;
    }

    Logger logger(tag, level, flags);
    if(loggers.append(logger))
    {
        return &loggers.last();
    }
    else
    {
        return nullptr;
    }
}

/*************************************************************************************************/
Logger* get(const char *tag)
{
    auto& loggers = get_loggers();

    for(auto& e : loggers) 
    {
        if(strcmp(tag, e.tag()) == 0)
        {
            return &e;
        }
    }

    return nullptr;
}

/*************************************************************************************************/
bool destroy(const char* tag)
{
    auto& loggers = get_loggers();

    const auto loggers_count = loggers.size();
    for(int i = 0; i < loggers_count; ++i) 
    {
        auto& e = loggers.get(i);
        if(strcmp(tag, e.tag()) == 0)
        {
            loggers.remove(i);
            return true;
        }
    }

    return false;
}

/*************************************************************************************************/
void debug(const char *tag, const char *fmt, ...)
{
    auto logger = get(tag);
    if(logger != nullptr) 
    {
        va_list args;
        va_start(args, fmt);
        logger->vwrite(Level::Debug, fmt, args);
        va_end(args);
    }
}

/************************************************************************************************/
void info(const char *tag, const char *fmt, ...)
{
    auto logger = get(tag);
    if(logger != nullptr) 
    {
        va_list args;
        va_start(args, fmt);
        logger->vwrite(Level::Info, fmt, args);
        va_end(args);
    }
}

/*************************************************************************************************/
void warn(const char *tag, const char *fmt, ...)
{
    auto logger = get(tag);
    if(logger != nullptr) 
    {
        va_list args;
        va_start(args, fmt);
        logger->vwrite(Level::Warn, fmt, args);
        va_end(args);
    }
}

/*************************************************************************************************/
void error(const char *tag, const char *fmt, ...)
{
    auto logger = get(tag);
    if(logger != nullptr) 
    {
        va_list args;
        va_start(args, fmt);
        logger->vwrite(Level::Error, fmt, args);
        va_end(args);
    }
}

/*************************************************************************************************/
void dump_hex(const char *tag, const void * address, unsigned length, const char *desc, ...)
{
    auto logger = get(tag);
    if(logger != nullptr) 
    {
        va_list args;
        va_start(args, desc);
        logger->vdump_hex(address, length, desc, args);
        va_end(args);
    }
}


} // namespace logging
