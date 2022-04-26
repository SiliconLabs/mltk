#pragma once


#include <cstdint>


#include "cpputils/helpers.hpp"




namespace cpputils 
{

/**
 * @brief Dictionary Item
 */
struct DictItem
{
    struct DictItem *next;
    const char* key;
    uint32_t length;
    void* value;

    template<typename T>
    T* as_type(void) const
    {
        return static_cast<T*>(value);
    }
};

/**
 * @brief Dictionary value iterator
 */
struct DictValueIterator
{
    DictItem *current;

    DictValueIterator(DictItem *head);

    bool operator!=(const DictValueIterator& rhs);
    void* operator*();
    void operator++();
};

/**
 * @brief Dictionary key iterator
 */
struct DictKeyIterator
{
    DictItem *current;

    DictKeyIterator(DictItem *head);

    bool operator!=(const DictKeyIterator& rhs);
    const char* operator*();
    void operator++();
};

/**
 * @brief Dictionary item iterator
 */
struct DictItemIterator
{
    DictItem *current;

    DictItemIterator(DictItem *head);

    bool operator!=(const DictItemIterator& rhs);
    DictItem* operator*();
    void operator++();
};


/**
 * @brief Dictionary Object
 */
class Dict
{


public:
    Dict() = default;
    ~Dict();

    bool put(const char* key, const void* value, uint32_t length, bool unique=true);

    template<typename T>
    bool put(const char* key, T* value, bool unique=true)
    {
        return put(key, (const void*)value, sizeof(T), unique);
    }

    bool update(const char* key, const void* value, uint32_t length);
    template<typename T>
    bool update(const char* key, const T* value)
    {
        return update(key, (const void*)value, sizeof(T));
    }


    void* get(const char* key) const;
    template<typename T>
    T* get(const char* key) const
    {
        return static_cast<T*>(get(key));
    }

    bool set(const char* key, const void* value, uint32_t length);
    template<typename T>
    bool set(const char* key, const T* value)
    {
        return set(key, (const void*)value, sizeof(T));
    }



    bool remove(const char* key);
    bool remove(const DictItem* item);
    bool contains(const char* key) const;
    void clear(void);

    void* operator [](const char* key) const
    {
        return get(key);
    }

    template<typename T>
    T* operator [](const char* key) const
    {
        return get<T>(key);
    }



    constexpr bool empty(void) const
    {
        return (count_ == 0);
    }

    constexpr uint32_t size(void) const
    {
        return count_;
    }

    constexpr DictItem* base(void) const
    {
        return items_;
    }

    DictValueIterator begin(void) const
    {
        return DictValueIterator(items_);
    }

    DictValueIterator end() const
    {
        return DictValueIterator(nullptr);
    }

    DictItemIterator items(void) const
    {
        return DictItemIterator(items_);
    }

    DictItemIterator enditems() const
    {
        return DictItemIterator(nullptr);
    }

    DictKeyIterator keys(void) const
    {
        return DictKeyIterator(items_);
    }

    DictKeyIterator endkeys() const
    {
        return DictKeyIterator(nullptr);
    }

protected:
    DictItem *items_ = nullptr;
    uint32_t count_ = 0;

    // Remove the copy constructors
    MAKE_CLASS_NON_ASSIGNABLE(Dict);


    friend DictItemIterator;
};



} // namespace cpputils 