#pragma once

#include <functional>

#include "cpputils/list.hpp"



namespace cpputils
{

/**
 * @brief Typed List Iterator
 */
template<typename T>
struct TypedListIterator
{
    uint8_t *ptr;
    const uint32_t entry_size;

    TypedListIterator(uint8_t *ptr, uint32_t entry_size) :
        ptr(ptr), entry_size(entry_size)
    {
    }


    bool operator!=(const TypedListIterator& rhs)
    {
        return ptr != rhs.ptr;
    }

    T& operator*()
    {
        return *reinterpret_cast<T*>(ptr);
    }

    void operator++()
    {
        ptr += entry_size;
    }
};


/**
 * @brief Typed List
 */
template<typename T>
class TypedList : public List
{
public:
    TypedList(uint32_t initial_capacity = 4) : List(sizeof(T), initial_capacity)
    {
    }

    TypedList(uint32_t max_capacity, void* buffer) : List(sizeof(T), max_capacity, buffer) 
    {
    }

    void initialize(uint32_t initial_capacity = 4)
    {
        List::initialize(sizeof(T), initial_capacity);
    }

    void initialize(uint32_t max_capacity, void* buffer) 
    {
        List::initialize(sizeof(T), max_capacity, buffer);
    }

    bool append(const T &val, bool unique=false)
    {
        return List::append(static_cast<const void*>(&val), unique);
    }

    bool prepend(const T &val, bool unique=false)
    {
        return List::prepend(static_cast<const void*>(&val), unique);
    }

    bool remove(const T &val)
    {
        return List::remove(static_cast<const void*>(&val));
    }

    bool remove(const std::function<bool(const T&)>& tester)
    {
        const int length = this->size();

        for(int i = 0; i < length; ++i)
        {
            if(tester(get(i)))
            {
                return List::remove(i);
            }
        }
        return false;
    }

    bool remove(int i)
    {
        return List::remove(i);
    }

    T& get(int i) const
    {
        return *reinterpret_cast<T*>(List::get(i));
    }

    bool contains(const T &val) const
    {
        return List::contains((void*)&val);
    }

    bool contains(const std::function<bool(const T&)>& tester) const
    {
        for(const auto& i : *this)
        {
            if(tester(i))
            {
                return true;
            }
        }
        return false;
    }

    constexpr T* list(void) const
    {
        return (T*)(head_);
    }

    T& operator [](int i) const
    {
        return get(i);
    }

    T& last() const
    {
        return get(size() - 1);
    }

    T& first() const
    {
        return get(0);
    }

    TypedListIterator<T> begin(void) const
    {
        return TypedListIterator<T>(head_, entry_size_);
    }

    TypedListIterator<T> end() const
    {
        return TypedListIterator<T>((head_ == nullptr) ? nullptr : &head_[count_*entry_size_], entry_size_);
    }
};

/**
 * @brief Static Typed List
 */
template<typename T, unsigned max_capacity>
class StaticTypedList : public TypedList<T>
{
public:
    StaticTypedList() : TypedList<T>(max_capacity, _buffer){}

private:
    uint8_t _buffer[sizeof(T) * max_capacity] alignas(T);
};


} // namespace cpputils
