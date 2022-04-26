#pragma once

#include <cstdint>

// These are defined in some C-lib headers
// Just undefine them for now
#undef major
#undef minor


namespace cpputils
{

/**
 * @brief Semantice Versioning Util
 * 
 * https://semver.org
 */
struct Semver
{
    static constexpr const unsigned INVALID = 0xFFFFFFFFUL;
    static constexpr const unsigned MAJOR_SHIFT = 24;
    static constexpr const unsigned MAJOR_MASK = 0xFF;
    static constexpr const unsigned MINOR_SHIFT = 16;
    static constexpr const unsigned MINOR_MASK = 0xFF;
    static constexpr const unsigned PATCH_SHIFT = 0;
    static constexpr const unsigned PATCH_MASK = 0xFFFF;

    uint32_t version;

    Semver(uint32_t version) : version(version){}

    static Semver parse(const char* version_str);
    static int compare(const Semver &a, const Semver &b);
    static int compare(const char* a, const char* b);
    int compare(const Semver &other) const;
    int compare(const uint32_t other) const;
    int compare(const char* other) const;

    static bool is_supported(const Semver &a, const Semver &b);
    static bool is_supported(const char* a, const char* b);
    bool is_supported(const Semver &other) const;
    bool is_supported(const uint32_t other) const;
    bool is_supported(const char* other) const;



    const char* to_str(char* buffer = nullptr) const;
    static const char* to_str(const Semver &ver, char* buffer = nullptr);
    static const char* to_str(uint32_t ver, char* buffer = nullptr);

    constexpr uint8_t major() const
    {
        return (version >> MAJOR_SHIFT) & MAJOR_MASK;
    }

    constexpr uint8_t minor() const
    {
        return (version >> MINOR_SHIFT) & MINOR_MASK;
    }

    constexpr uint16_t patch() const
    {
        return (version >> PATCH_SHIFT) & PATCH_MASK;
    }

    constexpr bool is_valid() const
    {
        return version != INVALID;
    }

};


} // namespace cpputils
