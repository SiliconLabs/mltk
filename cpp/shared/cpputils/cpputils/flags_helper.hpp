#pragma once


namespace cpputils 
{

/**
 * @brief Flags (i.e. bitfield) helper 
 */
template<typename FlagsType>
struct FlagsHelper
{
    FlagsType value;

    FlagsHelper(FlagsType value = (FlagsType)0) : value(value)
    {
    }

    void write(FlagsType v)
    {
        value = v;
    }

    constexpr FlagsType read() const
    {
        return value;
    }

    constexpr FlagsType read(FlagsType mask) const
    {
        return value & mask;
    }

    void set(FlagsType flags)
    {
        value |= flags;
    }

    void set(FlagsType flags, FlagsType mask)
    {
        value = (value & ~mask) | flags;
    }

    void clear(FlagsType flags)
    {
        value &= ~flags;
    }

    constexpr bool someSet(FlagsType flags) const
    {
        return (value & flags) != (FlagsType)0;
    }

    constexpr bool isSet(FlagsType flags) const
    {
        return (value & flags) != (FlagsType)0;
    }

    constexpr bool allSet(FlagsType flags) const
    {
        return (value & flags) == flags;
    }

    FlagsHelper(const FlagsHelper &other) : value(other.value)
    {
    }

    FlagsHelper& operator= (const FlagsHelper &other)
    {
        value = other.value;
        return *this;
    }

    FlagsHelper& operator= (const FlagsType type)
    {
        value = type;
        return *this;
    }


    FlagsHelper& operator |= (const FlagsHelper &other)
    {
        value |= other.value;
        return *this;
    }

    FlagsHelper& operator |= (const FlagsType type)
    {
        value |= type;
        return *this;
    }

    FlagsHelper& operator &= (const FlagsHelper &other)
    {
        value &= other.value;
        return *this;
    }

    FlagsHelper& operator &= (const FlagsType type)
    {
        value &= type;
        return *this;
    }

    FlagsHelper& operator ^= (const FlagsHelper &other)
    {
        value ^= other.value;
        return *this;
    }

    FlagsHelper& operator ^= (const FlagsType type)
    {
        value ^= type;
        return *this;
    }

    FlagsHelper operator| (const FlagsHelper &other) const
    {
        return value | other.value;
    }

    FlagsHelper operator| (const FlagsType type) const
    {
        return value | type;
    }

    FlagsHelper operator& (const FlagsHelper &other) const
    {
        return value & other.value;
    }

    FlagsHelper operator& (const FlagsType type) const
    {
        return value & type;
    }

    FlagsHelper operator^ (const FlagsHelper &other) const
    {
        return value ^ other.value;
    }

    FlagsHelper operator^ (const FlagsType type) const
    {
        return value ^ type;
    }

    FlagsHelper operator~ () const
    {
        return ~value;
    }

    FlagsHelper operator! () const
    {
        return !value;
    }

};



} // namespace cpputils
 