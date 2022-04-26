#include <cassert>
#include <cstdint>
#include <cstdlib>

#include "cpputils/buffer.hpp"


namespace cpputils 
{


/*************************************************************************************************/
Buffer::Buffer(unsigned size)
{
    if(size > 0)
    {
        init(size);
    }
}

/*************************************************************************************************/
Buffer::~Buffer()
{
    deinit();
}

/*************************************************************************************************/
void Buffer::init(unsigned size)
{
    _buffer = malloc(size);
    _size = (_buffer != nullptr) ? size : 0;
    _offset = 0;
}

/*************************************************************************************************/
void Buffer::deinit()
{
    _size = 0;
    _offset = 0;
    if(_buffer != nullptr)
    {
        free(_buffer);
        _buffer = nullptr;
    }
}

/*************************************************************************************************/
void* Buffer::data(unsigned offset)
{
    assert(offset < _size);
    return (_buffer != nullptr) ? static_cast<void*>(static_cast<uint8_t*>(_buffer) + offset) : nullptr;
}

/*************************************************************************************************/
void* Buffer::next(unsigned length)
{
    if(_offset + length > _size)
    {
        assert(!"Overflow");
        return nullptr;
    }
    
    void* retval = data(_offset);
    _offset += length;
    return retval;
}



} // namespace cpputils 
