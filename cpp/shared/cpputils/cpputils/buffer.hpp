
#pragma once


namespace cpputils 
{

/**
 * @brief Generic data buffer
 */
class Buffer
{
public:
    Buffer(unsigned size = 0);
    ~Buffer();

    void init(unsigned size);
    void deinit();

    void* data(unsigned offset = 0);
    void* next(unsigned length);

    template<typename T>
    T* data_as_type(unsigned offset = 0)
    {
        return static_cast<T*>(data(offset));
    }

    template<typename T>
    T* next_as_type(unsigned length)
    {
        return static_cast<T*>(next(length));
    }

    constexpr unsigned size() const 
    {
        return _size;
    }

    constexpr unsigned offset() const 
    {
        return _offset;
    }

    constexpr unsigned remaining() const 
    {
        return _size - _offset;
    }

    void reset_offset() 
    {
        _offset = 0;
    }

    bool isValid() const 
    {
        return _buffer != nullptr;
    }

protected:
    unsigned _size;
    void* _buffer;
    unsigned _offset;
};



} // namespace cpputils 
