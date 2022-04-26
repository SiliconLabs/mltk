

#include <cassert>
#include <cstring>
#include <cstdlib>


#include "cpputils/list.hpp"



namespace cpputils 
{


/*************************************************************************************************/
List::List(uint32_t entry_size, uint32_t initial_capacity)
{
    initialize(entry_size, initial_capacity);
}

/*************************************************************************************************/
List::List(uint32_t entry_size, uint32_t max_capacity, void* buffer)
{
    initialize(entry_size, max_capacity, buffer);
}


/*************************************************************************************************/
List::~List()
{
    clear();
}

/*************************************************************************************************/
void List::initialize(uint32_t entry_size, uint32_t initial_capacity)
{
    is_mutable_ = true;
    owns_buffer_ = true;
    head_ = nullptr;
    has_malloc_error_ = false;
    entry_size_ = entry_size;
    count_ = 0;
    initial_capacity_ = capacity_ = initial_capacity;
}

/*************************************************************************************************/
void List::initialize(uint32_t entry_size, uint32_t max_capacity, void* buffer)
{
    is_mutable_ = false;
    owns_buffer_ = false;
    has_malloc_error_ = false;
    head_ = static_cast<uint8_t*>(buffer);
    entry_size_ = entry_size;
    count_ = 0;
    initial_capacity_ = capacity_ = max_capacity;
}

/*************************************************************************************************/
bool List::clone(List &other)
{
    uint8_t* buffer = nullptr;

    if(count_ > 0)
    {
        buffer = static_cast<uint8_t*>(malloc(count_ * entry_size_));
        if(buffer == nullptr)
        {
            return false;
        }

        memcpy(buffer, head_, count_ * entry_size_);
    }

    other.is_mutable_ = is_mutable_;
    other.owns_buffer_ = true;
    other.has_malloc_error_ = false;
    other.head_ = buffer;
    other.entry_size_ = entry_size_;
    other.count_ = count_;
    other.initial_capacity_ = initial_capacity_;
    other.capacity_ = capacity_;

    return true;
}

/*************************************************************************************************/
bool List::append(const void* val, bool unique)
{
    bool retval = true;

    if(val == nullptr)
    {
        assert(!"null ptr");
        return false;
    }

    if(unique && contains(val))
    {
         goto exit;
    }

    if(head_ == nullptr)
    {
        retval = increase_size(capacity_);

        if(!retval)
        {
            goto exit;
        }
    }
    else if((uint32_t)count_ == capacity_)
    {
        retval = increase_size(capacity_*2);

        if(!retval)
        {
            goto exit;
        }
    }
    memcpy(&head_[count_*entry_size_], val, entry_size_);
    count_++;

    exit:
    return retval;
}

/*************************************************************************************************/
bool List::prepend(const void* val, bool unique)
{
    bool retval = true;

    if(val == nullptr)
    {
        assert(!"null ptr");
        return false;
    }

    if(unique && contains(val))
    {
         goto exit;
    }

    if(head_ == nullptr)
    {
        retval = increase_size(capacity_);

        if(!retval)
        {
            goto exit;
        }
    }
    else if((uint32_t)count_ == capacity_)
    {
        retval = increase_size(capacity_*2);

        if(!retval)
        {
            goto exit;
        }
    }
    memmove(&head_[1*entry_size_], &head_[0*entry_size_], count_*entry_size_);
    memcpy(&head_[0*entry_size_], val, entry_size_);
    count_++;

    exit:
    return retval;
}

/*************************************************************************************************/
bool List::remove(const void* val)
{
    bool retval = false;

    if(val == nullptr)
    {
        return false;
    }

    for(int i = 0; i < count_; ++i)
    {
        void *other_val = get(i);

        if(memcmp(val, other_val, entry_size_) == 0)
        {
            retval = true;

            memmove(&head_[i * entry_size_], &head_[((i + 1) * entry_size_)], (count_ - i - 1) * entry_size_);
            count_--;
            break;
        }
    }

    return retval;
}

/*************************************************************************************************/
bool List::remove(int i)
{
    if(i < 0 || i >= count_) 
    {
        return false;
    }

    memmove(&head_[i * entry_size_], &head_[((i + 1) * entry_size_)], (count_ - i - 1) * entry_size_);
    count_--;

    return true;
}

/*************************************************************************************************/
bool List::contains(const void *val) const
{
    bool retval = false;

    if(val == nullptr)
    {
        return false;
    }

    for(int i = 0; i < count_; ++i)
    {
        void *other_val = get(i);

        if(memcmp(val, other_val, entry_size_) == 0)
        {
            retval = true;
            break;
        }
    }

    return retval;
}

/*************************************************************************************************/
void* List::get(int i) const
{
    if(i < 0 || i >= count_)
    {
        assert(!"List index overflow");
        return nullptr;
    }

    return static_cast<void*>(&head_[i * entry_size_]);
}

/*************************************************************************************************/
bool List::increase_size(uint32_t new_size)
{
    uint8_t *old_list = head_;

    if(!is_mutable_)
    {
        assert(!"Buffer overflow");
        return false;
    }

    // If we're trying to malloc more than 1k, then just start incrementing rather than doubling
    if(new_size * entry_size_ > 1024) 
    {
        new_size = capacity_ + (256 + entry_size_ - 1) / entry_size_;
    }

    uint8_t* new_head = static_cast<uint8_t*>(malloc(new_size * entry_size_));
    if(new_head == nullptr)
    {
        has_malloc_error_ = true;
        return false;
    }

    head_ = new_head;

    if(old_list != nullptr)
    {
        memcpy(head_, old_list, capacity_ * entry_size_);
        free(old_list);
    }

    capacity_ = new_size;

    return true;
}


/*************************************************************************************************/
void List::clear(void)
{
    if(owns_buffer_ && head_ != nullptr)
    {
        free(head_);
        head_ = nullptr;
    }

    count_ = 0;
    capacity_ = initial_capacity_;
    has_malloc_error_ = false;
}

} // namespace cpputils 
