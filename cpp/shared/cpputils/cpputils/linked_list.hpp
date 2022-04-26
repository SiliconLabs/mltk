#pragma once


#include <cstdint>

#include "cpputils/helpers.hpp"


namespace cpputils 
{


struct LinkedListIterator;
class LinkedList;


/**
 * @brief  Linked List item
 */
struct LinkedListItem
{
    template<typename T>
    T* as_type(void) const
    {
        return reinterpret_cast<T*>(this);
    }

    virtual struct LinkedListItem* next() = 0;

    virtual void next(struct LinkedListItem* next) = 0;
    
protected:
    virtual void unlink(){};

    friend LinkedListIterator;
    friend LinkedList;
};

/**
 * @brief Linked List iterator
 */
struct LinkedListIterator
{
    LinkedListItem *current;

    LinkedListIterator(LinkedListItem *head) : current(head) {};

    bool operator!=(const LinkedListIterator& rhs)
    {
        return current != rhs.current;
    }

    void* operator*()
    {
        return (current == nullptr) ? nullptr : current;
    }

    void operator++()
    {
        if(current != nullptr)
        {
            current = current->next();
        }
    }
};



/**
 * @brief Linked List
 */
class LinkedList
{

public:

    LinkedList() = default;
    ~LinkedList();

    bool append(LinkedListItem *item);
    bool remove(LinkedListItem *item);
    bool remove(int i);


    LinkedListItem* get(int i) const;
    template<typename T>
    T* get(int i) const
    {
        return reinterpret_cast<T*>(LinkedList::get(i));
    }

    void clear(void);
    bool contains(const LinkedListItem *item) const;

    constexpr LinkedListItem* list(void) const
    {
        return head_;
    }

    template<typename T>
    constexpr T* list(void) const
    {
        return reinterpret_cast<T*>(head_);
    }

    constexpr uint32_t size(void) const
    {
        return count_;
    }

    constexpr bool empty(void) const
    {
        return count_ == 0;
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



    LinkedListIterator begin(void) const
    {
        return LinkedListIterator(head_);
    }

    LinkedListIterator end() const
    {
        return LinkedListIterator(nullptr);
    }

protected:
    LinkedListItem *head_ = nullptr;
    LinkedListItem *tail_ = nullptr;
    int count_ = 0;

    // Remove the copy constructors
    MAKE_CLASS_NON_ASSIGNABLE(LinkedList);

};



} // namespace cpputils
 