
#include <cstdio>
#include <ctype.h>


#include "cpputils/string.hpp"


namespace cpputils 
{



/*************************************************************************************************/
char* format_units(uint32_t number, uint8_t precision, char *buffer)
{
    static char default_buffer[24];
    const char *unit;
    uint32_t divisor;

    buffer = (buffer == nullptr) ? default_buffer : buffer;

    if(number > 1000*1000*1000)
    {
        unit = "G";
        divisor = 1000*1000*1000;
    }
    else if(number > 1000*1000)
    {
        unit = "M";
        divisor = 1000*1000;
    }
    else if(number > 1000)
    {
        unit = "k";
        divisor = 1000;
    }
    else
    {
        divisor = 1;
        unit = "";
    }

    if(divisor == 1)
    {
        sprintf(buffer, "%u", (unsigned int)number);
    }
    else
    {
        char fmt[10] = "%u.%0xu%s";

        fmt[5] = precision + '0';

        uint32_t precision_factor = 1;
        for(; precision > 0; --precision)
        {
            precision_factor *= 10;
        }

        const uint32_t scaled = ((uint64_t)number * precision_factor) / divisor;
        sprintf(buffer, fmt, scaled / precision_factor, scaled % precision_factor, unit);
    }

    return buffer;
}

/*************************************************************************************************/
const char* format_microseconds_to_milliseconds(uint32_t time_us, char *buffer)
{
    static char default_buffer[16];
    buffer = (buffer == nullptr) ? default_buffer : buffer;

    sprintf(buffer, "%3u.%03u ms", (unsigned int)(time_us / 1000), (unsigned int)(time_us % 1000));

    return buffer;
}

/*************************************************************************************************/
const char* format_rate(uint32_t total, uint32_t elapsed_time_us, uint8_t precision, char *buffer)
{
    if(elapsed_time_us > 0)
    {
        uint32_t metric_per_s = (((uint64_t)total * 1000000ULL) / elapsed_time_us);
        return format_units(metric_per_s, precision, buffer);
    }
    else
    {
        return "0";
    }
}


} //namespace cpputils



/*************************************************************************************************/
extern "C" int _strcasecmp(const char *s1, const char *s2)
{
  int d = 0;
  for ( ; ; )
    {
      const int c1 = tolower(*s1++);
      const int c2 = tolower(*s2++);
      if (((d = c1 - c2) != 0) || (c2 == '\0'))
        break;
    }
  return d;
}


/*************************************************************************************************/
#define IS_ALPHA(c) (((c) >= 'A' && (c) <= 'Z') || ((c) >= 'a' && (c) <= 'z'))
#define TO_UPPER(c) ((c) & 0xDF)
extern "C" const char* _strcasestr(const char *str1, const char *str2)
{
    const char *cp = str1;
    const char *s1, *s2;

    if ( !*str2 )
    {
        return str1;
    }
        

    while (*cp)
    {
        s1 = cp;
        s2 = str2;

        while ( *s1 && *s2 && (IS_ALPHA(*s1) && IS_ALPHA(*s2)) ? !(TO_UPPER(*s1) - TO_UPPER(*s2)):!(*s1-*s2))
        {
            ++s1, ++s2;
        } 

        if (!*s2)
        {
            return cp;
        }
               
        ++cp;
    }

    return nullptr;
}