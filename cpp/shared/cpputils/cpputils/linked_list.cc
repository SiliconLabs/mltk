
#include <cstring>
#include <cstdlib>


#include "cpputils/linked_list.hpp"


namespace cpputils 
{



/*************************************************************************************************/
LinkedList::~LinkedList()
{
    clear();
}

/*************************************************************************************************/
bool LinkedList::append(LinkedListItem *item)
{
    if(head_ == nullptr)
    {
        head_ = tail_ = item;
    }
    else
    {
        tail_->next(item);
        tail_ = item;
    }
    count_++;

    return true;
}

/*************************************************************************************************/
LinkedListItem* LinkedList::get(int i) const
{
    LinkedListItem* retval = nullptr;

    if(i >= count_)
    {
        return nullptr;
    }

    int index = 0;
    for(auto item = head_; item != nullptr; ++index, item = item->next())
    {
        if(index == i)
        {
            retval = item;
            break;
        }
    }

    return retval;
}


/*************************************************************************************************/
bool LinkedList::remove(LinkedListItem* value)
{
    bool retval = false;

    if(value == nullptr)
    {
        return false;
    }

    LinkedListItem *prev = nullptr;

    for(auto item = head_; item != nullptr; prev = item, item = item->next())
    {
        if(item == value)
        {
            if(prev == nullptr)
            {
                head_ = item->next();
            }
            else
            {
                prev->next(item->next());
            }

            if(tail_ == item)
            {
                tail_ = prev;
            }

            item->next(nullptr);
            item->unlink();
            count_--;
            retval = true;

            break;
        }
    }

    return retval;
}


/*************************************************************************************************/
bool LinkedList::remove(int i)
{
    bool retval = false;

    if(i >= count_)
    {
        return false;
    }

    LinkedListItem *prev = nullptr;
    int index = 0;
    for(auto item = head_; item != nullptr; ++index, prev = item, item = item->next())
    {
        if(index == i)
        {
            if(prev == nullptr)
            {
                head_ = item->next();
            }
            else
            {
                prev->next(item->next());
            }

            if(tail_ == item)
            {
                tail_ = prev;
            }

            item->next(nullptr);
            item->unlink();
            count_--;
            retval = true;

            break;
        }
    }

    return retval;
}

/*************************************************************************************************/
bool LinkedList::contains(const LinkedListItem *value) const
{
    bool retval = false;

    if(value == nullptr)
    {
        return false;
    }


    for(auto item = head_; item != nullptr; item = item->next())
    {
        if(item == value)
        {
            retval = true;
            break;
        }
    }

    return retval;
}

/*************************************************************************************************/
void LinkedList::clear(void)
{
    for(auto next = head_; next != nullptr; )
    {
        auto current = next;
        next = current->next();
        current->next(nullptr);
        current->unlink();
    }

    count_ = 0;
    head_ = nullptr;
    tail_ = nullptr;
}



} // namespace cpputils 