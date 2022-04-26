
#pragma once

#include <cstdint>
#include <cstdbool>

#include "cpputils/helpers.hpp"

#ifdef __cplusplus
extern "C" {
#endif




#ifndef HEAP_MALLOC
#define HEAP_MALLOC(size) heap_malloc(size)
#endif

#ifndef HEAP_MALLOC_ALIGNED
#define HEAP_MALLOC_ALIGNED(size, align) heap_malloc_aligned(size, align)
#endif

#ifndef HEAP_MALLOC_OBJECT
#define HEAP_MALLOC_OBJECT(type) (type*)heap_malloc_aligned(sizeof(type), alignof(type))
#endif

#ifndef HEAP_FREE
#define HEAP_FREE(p) heap_free((void*)p)
#endif


struct HeapStats
{
    uint32_t used;
    uint32_t remaining;
    uint32_t size;
};

DLL_EXPORT void* heap_malloc_aligned(uint32_t size, uint32_t alignment);
DLL_EXPORT void* heap_malloc(uint32_t size); 
DLL_EXPORT void heap_free(void* ptr);

DLL_EXPORT void heap_set_buffer(void* buffer, uint32_t length);
DLL_EXPORT bool heap_get_buffer(void** buffer_ptr);
DLL_EXPORT bool heap_get_stats(HeapStats *stats);



#ifdef __cplusplus
}
#endif

