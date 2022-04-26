
#pragma once


#include <cstdint>
#include <cstring>

namespace cpputils
{

char* format_units(uint32_t number, uint8_t precision = 2, char *buffer = nullptr);
const char* format_microseconds_to_milliseconds(uint32_t time_us, char *buffer = nullptr);
const char* format_rate(uint32_t total, uint32_t elapsed_time_us, uint8_t precision = 2, char *buffer = nullptr);

} // namespace cpputils

#undef strcasecmp
#define strcasecmp(s1, s2) _strcasecmp(s1, s2)
extern "C" int _strcasecmp(const char *s1, const char *s2);

#undef strcasestr
#define strcasestr(s1, s2) _strcasestr(s1, s2)
extern "C" const char* _strcasestr(const char *s1, const char *s2);
