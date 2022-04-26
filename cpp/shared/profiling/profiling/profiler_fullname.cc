#include <cstring>

#include "profiling/profiler.hpp"

namespace profiling
{

const char Fullname::SEPARATOR[] = "::";


/*************************************************************************************************/
bool Fullname::create(Fullname& fullname, const char* name, Profiler* parent)
{
    if(name == nullptr || *name == 0)
    {
        fullname._invalid = true;
        return false;
    }

    if(parent != nullptr)
    {
        parent->fullname(fullname);
    }

    fullname.append(Fullname::SEPARATOR);
    fullname.append(name);

    return fullname.is_valid();
}

/*************************************************************************************************/
void Fullname::append(const char* name)
{
    if(_invalid)
    {
        return;
    }

    if(ptr == nullptr)
    {
        ptr = value;
    }

    const auto remaining_space = (unsigned)(&value[sizeof(value)] - ptr);
    const auto len = strlen(name);

    if(len >= remaining_space)
    {
        _invalid = true;
        return;
    }

    strcat(ptr, name);
    ptr += len;
}


  
} // namespace profiling
