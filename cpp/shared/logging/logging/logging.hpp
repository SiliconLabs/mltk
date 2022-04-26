#pragma once

#include <cstring>
#include <stdint.h>
#include <stdarg.h>
#include <functional>

#include "cpputils/typed_list.hpp"
#include "cpputils/helpers.hpp"
#include "cpputils/flags_helper.hpp"



namespace logging
{

class Logger;
using LoggerList = cpputils::TypedList<Logger>;

constexpr const unsigned MAX_TAG_LENGTH = 15;
constexpr const unsigned MAX_FMT_LENGTH = 256;

enum class Level : uint8_t
{
    Debug,
    Info,
    Warn,
    Error,
    Disabled,
    Count
};
constexpr Level Debug = Level::Debug;
constexpr Level Info = Level::Info;
constexpr Level Warn = Level::Warn;
constexpr Level Error = Level::Error;
constexpr Level Disabled = Level::Disabled;


enum class Flag : uint8_t
{
    None        = 0,
    PrintTag   = (1 << 0),
    PrintLevel = (1 << 1),
    Newline     = (1 << 2)
};
DEFINE_ENUM_CLASS_BITMASK_OPERATORS(Flag, uint8_t)

constexpr Flag None = Flag::None;
constexpr Flag PrintTag = Flag::PrintTag;
constexpr Flag PrintLevel = Flag::PrintLevel;
constexpr Flag Newline = Flag::Newline;



typedef cpputils::FlagsHelper<Flag> Flags;

using Writer = std::function<void(const char*msg, int length, void* arg)>;

void set_logger_list(LoggerList* list);
DLL_EXPORT LoggerList& get_loggers();

Logger* create(const char* tag, Level level = Level::Info, Flags flags = Flag::Newline);
Logger* get(const char *tag);
bool destroy(const char* tag);


void debug(const char *tag, const char *fmt, ...);
void info(const char *tag, const char *fmt, ...);
void warn(const char *tag, const char *fmt, ...);
void error(const char *tag, const char *fmt, ...);
void dump_hex(const char *tag, const void * address, unsigned length, const char *desc=nullptr, ...);

} // namespace logging

