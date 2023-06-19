#include <string>
#include <cstdarg>


namespace cpputils
{

std::string vformat(const char* fmt_str, va_list args);
std::string format(const char* fmt_str, ...);

} // namespace cpputils
