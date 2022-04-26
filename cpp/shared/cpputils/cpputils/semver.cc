#include <cstring>
#include <cstdio>
#include <cstdlib>

#include "cpputils/semver.hpp"


extern "C" char *strtok_r(char *, const char *, char **);

namespace cpputils
{




/*************************************************************************************************/
Semver Semver::parse(const char* version_str)
{
    static const uint8_t offsets[] =
    {
     Semver::MAJOR_SHIFT,
     Semver::MINOR_SHIFT,
     Semver::PATCH_SHIFT,
    };
    static const uint16_t masks[] =
    {
     (uint16_t)(~Semver::MAJOR_MASK),
     (uint16_t)(~Semver::MINOR_MASK),
     (uint16_t)(~Semver::PATCH_MASK),
    };


    char *tok, *ptr;
    uint32_t version = 0;
    const char decimal_str[2] = {'.', 0 };
    char buffer[20];

    strncpy(buffer, version_str, sizeof(buffer)-1);
    buffer[sizeof(buffer)-1] = 0;
    ptr = buffer;


    for(int i = 0; i < 4 && (tok = strtok_r(ptr, decimal_str, &ptr)) != NULL; ++i)
    {
        char *end;
        if(i == 3)
        {
            version = Semver::INVALID;
            goto exit;
        }

        const uint32_t value = strtol(tok, &end, 10);
        if(*end != 0)
        {
            version = Semver::INVALID;
            goto exit;
        }
        else if(value & masks[i])
        {
            version = Semver::INVALID;
            goto exit;
        }
        version |= (value << offsets[i]);
    }

    exit:
    return Semver(version);
}

/*************************************************************************************************/
int Semver::compare(const Semver &a, const Semver &b)
{
    if(a.is_valid() || b.is_valid())
    {
        return INT32_MIN;
    }
    else if(a.major() > b.major())
    {
        return 1;
    }
    else if(a.major() < b.major())
    {
        return -1;
    }
    else if(a.minor() > b.minor())
    {
        return 1;
    }
    else if(a.minor() < b.minor())
    {
        return -1;
    }
    else if(a.patch() > b.patch())
    {
        return 1;
    }
    else if(a.patch() < b.patch())
    {
        return -1;
    }
    else
    {
        return 0;
    }
}

/*************************************************************************************************/
int Semver::compare(const char* a, const char* b)
{
    return Semver::compare(Semver::parse(a), Semver::parse(b));
}

/*************************************************************************************************/
int Semver::compare(const Semver &other) const
{
    return Semver::compare(*this, other);
}

/*************************************************************************************************/
int Semver::compare(const uint32_t other) const
{
    return Semver::compare(*this, Semver(other));
}

/*************************************************************************************************/
int Semver::compare(const char* other) const
{
    return Semver::compare(*this, Semver::parse(other));
}

/*************************************************************************************************/
bool Semver::is_supported(const Semver &a, const Semver &b)
{
    return a.major() == b.major();
}

/*************************************************************************************************/
bool Semver::is_supported(const char* a, const char* b)
{
    return Semver::is_supported(Semver::parse(a), Semver::parse(b));
}

/*************************************************************************************************/
bool Semver::is_supported(const Semver &other) const
{
    return Semver::is_supported(*this, other);
}

/*************************************************************************************************/
bool Semver::is_supported(const uint32_t other) const
{
    return Semver::is_supported(*this, Semver(other));
}

/*************************************************************************************************/
bool Semver::is_supported(const char* other) const
{
    return Semver::is_supported(*this, Semver::parse(other));
}




/*************************************************************************************************/
const char* Semver::to_str(char* buffer) const
{
    return Semver::to_str(*this, buffer);
}

/*************************************************************************************************/
const char* Semver::to_str(uint32_t ver, char* buffer)
{
    return Semver::to_str(Semver(ver), buffer);
}

/*************************************************************************************************/
const char* Semver::to_str(const Semver &ver, char* buffer)
{
    static char local_buffer[20];

    buffer = (buffer == nullptr) ? local_buffer: buffer;

    if(ver.is_valid())
    {
        strcpy(buffer, "invalid");
    }
    else
    {
        snprintf(buffer, 19, "%u.%u.%u", ver.major(), ver.minor(), ver.patch());
    }

    return buffer;
}


} //namespace cpputils
