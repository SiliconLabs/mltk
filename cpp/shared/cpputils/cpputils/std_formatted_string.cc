#include <string>
#include <cstring>
#include <memory>
#include <cstdarg>



#include "std_formatted_string.hpp"


namespace cpputils
{



std::string vformat(const char* fmt_str, va_list args)
{
    int final_n, n = 256;
    std::unique_ptr<char[]> formatted;

    while(1)
    {
        formatted.reset(new char[n]); /* Wrap the plain char array into the unique_ptr */
        strcpy(&formatted[0], fmt_str);

        final_n = vsnprintf(formatted.get(), n, fmt_str, args);

        if (final_n < 0 || final_n >= n)
        {
            n += abs(final_n - n + 1);
        }
        else
        {
            break;
        }
    }

    return std::string(formatted.get());
}

std::string format(const char* fmt_str, ...)
{
    va_list args;

    va_start(args, fmt_str);
    auto retval = vformat(fmt_str, args);
    va_end(args);

    return retval;
}




} // namespace cpputils
