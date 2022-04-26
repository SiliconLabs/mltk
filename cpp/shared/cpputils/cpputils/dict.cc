#include <cstring>
#include <cstdlib>


#include "cpputils/dict.hpp"



namespace  cpputils
{
    

/*************************************************************************************************/
Dict::~Dict()
{
    clear();
}

/*************************************************************************************************/
bool Dict::put(const char* key, const void* value, uint32_t length, bool unique)
{
    if(unique && contains(key))
    {
        return false;
    }

    uint8_t* ptr = static_cast<uint8_t*>(malloc(sizeof(DictItem) + strlen(key) + 1 + length));

    if(ptr == nullptr)
    {
        return false;
    }

    auto item = reinterpret_cast<DictItem*>(ptr);
    ptr += sizeof(DictItem);
    item->value = ptr;
    item->length = length;
    ptr += length;
    item->key = (char*)ptr;

    strcpy((char*)item->key, key);
    memcpy(item->value, value, length);

    item->next = items_;
    items_ = item;

    count_++;

    return true;
}


/*************************************************************************************************/
void* Dict::get(const char* key) const
{
    void* retval = nullptr;

    for(auto item = items_; item != nullptr; item = item->next)
    {
        if(strcmp(key, item->key) == 0)
        {
            retval = item->value;
            break;
        }
    }

    return retval;
}

/*************************************************************************************************/
bool Dict::set(const char* key, const void* value, uint32_t length)
{
    if(contains(key))
    {
        return update(key, value, length);
    }
    else
    {
        return put(key, value, length);
    }
}

/*************************************************************************************************/
bool Dict::update(const char* key, const void* value, uint32_t length)
{
    bool retval = false;

    DictItem *prev = nullptr;
    for(auto item = items_; item != nullptr; prev = item, item = item->next)
    {
        if(strcmp(item->key, key) == 0)
        {
            if(length == item->length)
            {
                memcpy(item->value, value, length);
                retval = true;
                break;
            }

            uint8_t* ptr = static_cast<uint8_t*>(malloc(sizeof(DictItem) + strlen(key) + 1 + length));

            if(ptr == nullptr)
            {
                retval = false;
                break;
            }

            auto new_item = reinterpret_cast<DictItem*>(ptr);
            ptr += sizeof(DictItem);
            new_item->value = ptr;
            ptr += length;
            new_item->key = (char*)ptr;

            strcpy((char*)new_item->key, key);
            memcpy(new_item->value, value, length);

            if(prev != nullptr)
            {
                prev->next = new_item;
            }
            new_item->next = item->next;
            free(item);
            retval = true;
            break;
        }
    }

    return retval;
}


/*************************************************************************************************/
bool Dict::remove(const char* key)
{
    bool retval = false;

    DictItem *prev = nullptr;
    for(auto item = items_; item != nullptr; prev = item, item = item->next)
    {
        if(strcmp(item->key, key) == 0)
        {
            if(prev == nullptr)
            {
                items_ = item->next;
            }
            else
            {
                prev->next = item->next;
            }

            retval = true;
            count_--;

            free(item);

            break;
        }
    }

    return retval;
}

/*************************************************************************************************/
bool Dict::remove(const DictItem* needle)
{
    bool retval = false;

    DictItem *prev = nullptr;
    for(auto item = items_; item != nullptr; prev = item, item = item->next)
    {
        if(needle == item)
        {
            if(prev == nullptr)
            {
                items_ = item->next;
            }
            else
            {
                prev->next = item->next;
            }

            retval = true;
            count_--;

            free(item);

            break;
        }
    }

    return retval;
}

/*************************************************************************************************/
bool Dict::contains(const char* key) const
{
    bool retval = false;

    for(auto item = items_; item != nullptr; item = item->next)
    {
        if(strcmp(key, item->key) == 0)
        {
            retval = true;
            break;
        }
    }

    return retval;
}

/*************************************************************************************************/
void Dict::clear(void)
{
    for(auto next = items_; next != nullptr; )
    {
        auto current = next;
        next = current->next;
        free(current);
    }

    count_ = 0;
    items_ = nullptr;
}



/*************************************************************************************************/
DictValueIterator::DictValueIterator(DictItem *head) :
        current(head)
{
};

/*************************************************************************************************/
bool DictValueIterator::operator!=(const DictValueIterator& rhs)
{
    return current != rhs.current;
}

/*************************************************************************************************/
void* DictValueIterator::operator*()
{
    return (current == nullptr) ? nullptr : current->value;
}

/*************************************************************************************************/
void DictValueIterator::operator++()
{
    if(current != nullptr)
    {
        current = current->next;
    }
}


/*************************************************************************************************/
DictKeyIterator::DictKeyIterator(DictItem *head) :
        current(head)
{
};

/*************************************************************************************************/
bool DictKeyIterator::operator!=(const DictKeyIterator& rhs)
{
    return current != rhs.current;
}

/*************************************************************************************************/
const char* DictKeyIterator::operator*()
{
    return (current == nullptr) ? nullptr : current->key;
}

/*************************************************************************************************/
void DictKeyIterator::operator++()
{
    if(current != nullptr)
    {
        current = current->next;
    }
}

/*************************************************************************************************/
DictItemIterator::DictItemIterator(DictItem *head) :
        current(head)
{
};

/*************************************************************************************************/
bool DictItemIterator::operator!=(const DictItemIterator& rhs)
{
    return current != rhs.current;
}

/*************************************************************************************************/
DictItem* DictItemIterator::operator*()
{
    return current;
}

/*************************************************************************************************/
void DictItemIterator::operator++()
{
    if(current != nullptr)
    {
        current = current->next;
    }
}



} // namespace cpputils
