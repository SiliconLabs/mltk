#pragma once



#include "cpputils/dict.hpp"


namespace cpputils
{

/**
 * @brief Typed Dictionary Item
 */
template<typename T>
struct TypedDictItem
{
    struct TypedDictItem<T> *next;
    const char* key;
    uint32_t length;
    T* value;
};

template<typename T>
struct TypedDictValueIterator
{
    TypedDictItem<T> *current;
    TypedDictValueIterator(TypedDictItem<T> *head) : current(head){}
    bool operator!=(const TypedDictValueIterator& rhs) { return current != rhs.current; }
    T* operator*(){ return (current == nullptr) ? nullptr : current->value; }
    void operator++() { if(current != nullptr) current = current->next; }
};

/**
 * @brief Typed Dictionary Key Iterator
 */
template<typename T>
struct TypedDictKeyIterator
{
    TypedDictItem<T> *current;
    TypedDictKeyIterator(TypedDictItem<T> *head) : current(head){}
    bool operator!=(const TypedDictKeyIterator& rhs) { return current != rhs.current; }
    const char* operator*() { return (current == nullptr) ? nullptr : current->key; }
    void operator++() { if(current != nullptr) current = current->next; }
};

/**
 * @brief Typed Dictionary Item Iterator
 */
template<typename T>
struct TypedDictItemIterator
{
    TypedDictItem<T> *current;
    TypedDictItemIterator(TypedDictItem<T> *head) : current(head){}
    bool operator!=(const TypedDictItemIterator& rhs){ return current != rhs.current; }
    TypedDictItem<T>* operator*(){ return current; }
    void operator++(){  if(current != nullptr) current = current->next; }
};

/**
 * @brief Typed Dictionary
 */
template<typename T>
class TypedDict : public Dict
{
public:
    bool put(const char* key, const T* value, bool unique=true)
    {
        return Dict::put(key, (const void*)value, sizeof(T), unique);
    }

    bool update(const char* key, const T* value)
    {
        return Dict::update(key, (const void*)value, sizeof(T));
    }

    T* get(const char* key) const
    {
        return static_cast<T*>(Dict::get(key));
    }

    bool set(const char* key, const T* value)
    {
        return Dict::set(key, (const void*)value, sizeof(T));
    }

    bool remove(const TypedDictItem<T>* item)
    {
        return Dict::remove(reinterpret_cast<const DictItem*>(item));
    }

    T* operator [](const char* key) const
    {
        return get(key);
    }

    constexpr TypedDictItem<T>* base(void) const
    {
        return reinterpret_cast<TypedDictItem<T>*>(items_);
    }

    TypedDictValueIterator<T> begin(void) const
    {
        return TypedDictValueIterator<T>((TypedDictItem<T>*)items_);
    }

    TypedDictValueIterator<T> end() const
    {
        return TypedDictValueIterator<T>(nullptr);
    }

    TypedDictItemIterator<T> items(void) const
    {
        return TypedDictItemIterator<T>((TypedDictItem<T>*)items_);
    }

    TypedDictItemIterator<T> enditems() const
    {
        return TypedDictItemIterator<T>(nullptr);
    }

    TypedDictKeyIterator<T> keys(void) const
    {
        return TypedDictKeyIterator<T>((TypedDictItem<T>*)items_);
    }

    TypedDictKeyIterator<T> endkeys() const
    {
        return TypedDictKeyIterator<T>(nullptr);
    }
};



} // namespace cpputils
