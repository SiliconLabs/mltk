#pragma once

#include <functional>

#include "cpputils/linked_list.hpp"



namespace cpputils
{

template<typename T>
struct TypedLinkedListIterator
{
    LinkedListItem *current;

    TypedLinkedListIterator(LinkedListItem *head) : current(head) {};

    bool operator!=(const TypedLinkedListIterator<T>& rhs)
    {
        return current != rhs.current;
    }

    T* operator*()
    {
        return reinterpret_cast<T*>(current);
    }

    void operator++()
    {
        if(current != nullptr)
        {
            current = current->next();
        }
    }
};


template<typename T>
struct TypedLinkedList : public LinkedList
{
    bool remove(T* item)
    {
        return LinkedList::remove(reinterpret_cast<LinkedListItem*>(item));
    }

    bool remove(const std::function<bool(const T*)>& tester)
    {
        const int length = this->size();

        for(int i = 0; i < length; ++i)
        {
            if(tester(get(i)))
            {
                return LinkedList::remove(i);
            }
        }
        return false;
    }

    T* get(int i) const
    {
        return LinkedList::get<T>(i);
    }

    bool contains(const T* item)
    {
        return LinkedList::contains(reinterpret_cast<const LinkedListItem*>(item));
    }

    bool contains(const std::function<bool(const T*)>& tester) const
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
        return LinkedList::list<T>();
    }

    T* operator [](int i) const
    {
        return LinkedList::get<T>(i);
    }

    TypedLinkedListIterator<T> begin(void) const
    {
        return TypedLinkedListIterator<T>(head_);
    }

    TypedLinkedListIterator<T> end() const
    {
        return TypedLinkedListIterator<T>(nullptr);
    }

};



} // namespace cpputils
