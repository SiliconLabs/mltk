#pragma once



#include <cstdint>


#include "cpputils/helpers.hpp"



namespace cpputils 
{


/**
 * @brief List Iterator
 */
struct ListIterator
{
    uint8_t *ptr;
    uint32_t entry_size;

    ListIterator(uint8_t *ptr, uint32_t entry_size) :
        ptr(ptr), entry_size(entry_size)
    {
    }


    bool operator!=(const ListIterator& rhs)
    {
        return ptr != rhs.ptr;
    }

    void* operator*()
    {
        return static_cast<void*>(ptr);
    }

    void operator++()
    {
        ptr += entry_size;
    }
};




/**
 * @brief List object
 */
class List
{

public:

    List(uint32_t entry_size=sizeof(void*), uint32_t initial_capacity = 4);
    List(uint32_t entry_size, uint32_t max_capacity, void* buffer);
    ~List();

    void initialize(uint32_t entry_size=sizeof(void*), uint32_t initial_capacity = 4);
    void initialize(uint32_t entry_size, uint32_t max_capacity, void* buffer);

    bool clone(List &other);

    bool append(const void* val, bool unique=false);
    bool prepend(const void* val, bool unique=false);
    bool remove(const void* val);
    bool remove(int i);
    void* get(int i) const;
    template<typename T>
    T* get(int i) const
    {
        return static_cast<T*>(get(i));
    }

    void clear(void);
    bool contains(const void *val) const;

    constexpr void* list(void) const
    {
        return static_cast<void*>(head_);
    }


    void* operator [](int i) const
    {
        return get(i);
    }

    template<typename T>
    T* operator [](int i) const
    {
        return get<T>(i);
    }


    constexpr uint32_t size(void) const
    {
        return count_;
    }

    constexpr uint32_t capacity(void) const
    {
        return capacity_;
    }

    constexpr bool empty(void) const
    {
        return count_ == 0;
    }

    constexpr bool full(void) const
    {
        return count_ == capacity_;
    }

    ListIterator begin(void) const
    {
        return ListIterator(head_, entry_size_);
    }

    ListIterator end() const
    {
        return ListIterator((head_ == nullptr) ? nullptr : &head_[count_*entry_size_], entry_size_);
    }

    bool has_malloc_error() const 
    {
        return has_malloc_error_;
    }

protected:
    uint8_t *head_;
    uint32_t entry_size_;
    int32_t count_;
    uint32_t initial_capacity_;
    uint32_t capacity_;
    bool is_mutable_;
    bool has_malloc_error_;
    bool owns_buffer_;

    bool increase_size(uint32_t new_size);


    // Remove the copy constructors
    MAKE_CLASS_NON_ASSIGNABLE(List);
};



} // namespace cpputils 