#pragma once


#include <initializer_list>


namespace cpputils 
{


/**
 * @brief Provides [] for initializer_list
 */
template<class T>
struct _initializer_list_with_square_brackets
{
    const std::initializer_list<T>& list;

    _initializer_list_with_square_brackets(const std::initializer_list<T>& _list): list(_list) {}

    const T& operator[](unsigned int index) const
    {
        return *(list.begin() + index);
    }
};

/**
 * @brief Provides [] for initializer_list
 * @return const initializer_list_with_square_brackets<T> 
 */
template<class T>
const _initializer_list_with_square_brackets<T> initializer_list_accessor(const std::initializer_list<T>& list) {
    return _initializer_list_with_square_brackets<T>(list);
}

/**
 * @brief Add bitwise operations to enum class datatype
 */
#define DEFINE_ENUM_CLASS_BITMASK_OPERATORS(E, T)\
    inline E operator | (E lhs, E rhs){\
        return E(T(lhs) | T(rhs));\
    }\
    inline E operator & (E lhs, E rhs){\
        return E(T(lhs) & T(rhs));\
    }\
    inline E operator ^ (E lhs, E rhs){\
        return E(T(lhs) ^ T(rhs));\
    }\
    inline E operator ~ (E lhs){\
        return E(~T(lhs));\
    }\
    inline E operator |= (E &lhs, E rhs){\
        lhs = lhs | rhs;\
        return lhs;\
    }\
    inline E operator &= (E &lhs, E rhs){\
        lhs = lhs & rhs;\
        return lhs;\
    }\
    inline E operator ^= (E &lhs, E rhs){\
        lhs = lhs ^ rhs;\
        return lhs;\
    }\
    inline bool operator !(E rhs) \
    {\
        return !bool(T(rhs)); \
    }



static inline unsigned int align_up(int value, unsigned align_by)
{
    return ((value + align_by - 1) / align_by) * align_by;
}

static inline unsigned int align_down(int value, unsigned divide_by)
{
    return (value / divide_by) * divide_by;
}


#define VARGS_0(x, ...) x
#define VARGS_1(x, ...) VARGS_0(__VA_ARGS__) 
#define VARGS_2(x, ...) VARGS_1(__VA_ARGS__) 
#define VARGS_3(x, ...) VARGS_2(__VA_ARGS__) 
#define VARGS_4(x, ...) VARGS_3(__VA_ARGS__) 
#define VARGS_5(x, ...) VARGS_4(__VA_ARGS__) 
#define VARGS_6(x, ...) VARGS_5(__VA_ARGS__)
#define VARGS_7(x, ...) VARGS_6(__VA_ARGS__)
#define VARGS_8(x, ...) VARGS_7(__VA_ARGS__)
#define VARGS_9(x, ...) VARGS_8(__VA_ARGS__)
#define VARGS_10(x, ...) VARGS_9(__VA_ARGS__)
#define VARGS_11(x, ...) VARGS_10(__VA_ARGS__)
#define VARGS_12(x, ...) VARGS_11(__VA_ARGS__)
#define VARGS_13(x, ...) VARGS_12(__VA_ARGS__)
#define VARGS_14(x, ...) VARGS_13(__VA_ARGS__)
#define VARGS_15(x, ...) VARGS_14(__VA_ARGS__)



#ifndef MIN
#define MIN(x,y)  ((x) < (y) ? (x) : (y))
#endif /* ifndef MIN */
#ifndef MAX
#define MAX(x,y)  ((x) > (y) ? (x) : (y))
#endif /* ifndef MAX */

#define ARRAY_COUNT(x) (sizeof (x) / sizeof *(x))
#define ALIGN_n(x, n) ((((uint32_t)x) + ((n)-1)) & ~((n)-1))
#define ALIGN_32(x) ALIGN_n(x, 32)
#define ALIGN_16(x) ALIGN_n(x, 16)
#define ALIGN_8(x) ALIGN_n(x, 8)
#define ALIGN_4(x) ALIGN_n(x, 4)


#define MAKE_CLASS_NON_ASSIGNABLE(name) \
name(const name &other) = delete; \
name(name &other) = delete; \
name(name &&other) = delete; \
name& operator= (const name &other) = delete;

#define CREATE_STATIC_OBJECT_BUFFER(T, name) alignas(T) static uint8_t name[sizeof(T)]


#ifndef STRINGIFY
// Note: The C preprocessor stringify operator ('#') makes a string from its argument, without macro expansion
// e.g. If "version" is #define'd to be "4", then STRINGIFY_AWE(version) will return the string "version", not "4"
// To expand "version" to its value before making the string, use STRINGIFY(version) instead
#define STRINGIFY_ARGUMENT_WITHOUT_EXPANSION(s) #s
#define STRINGIFY(s) STRINGIFY_ARGUMENT_WITHOUT_EXPANSION(s)
#endif


#if defined(MLTK_DLL_EXPORT) && !defined(DLL_EXPORT)
#  if defined(WIN32) || defined(_WIN32)
#    define DLL_EXPORT __declspec(dllexport)
#  else
#    define DLL_EXPORT __attribute__ ((visibility ("default")))
#  endif
#endif 

#ifndef DLL_EXPORT
#define DLL_EXPORT
#endif




} // namespace cpputils 
