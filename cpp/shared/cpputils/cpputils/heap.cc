#ifndef MLTK_DLL_IMPORT

#include <cstring>
#include <cassert>


#include "cpputils/heap.hpp"
#include "cpputils/helpers.hpp"


using namespace cpputils;


#if INTPTR_MAX == INT64_MAX
#define STRUCT_PADDING uint8_t padding[4];
#else
#define STRUCT_PADDING
#endif
#ifdef __arm__
#define acquire_lock()
#define release_lock()
#else
void acquire_lock();
void release_lock();
#endif

#pragma pack(1)
struct pool;
typedef struct header
{
    struct header *ptr;         /* Pointer to circular successor */
    uint32_t size;              /* Size in units of blocks */
    STRUCT_PADDING
} header_t;

typedef struct pool
{
    header_t base;
    header_t *freep;            /* Current entry point to free list */
    uint32_t total_size;
    STRUCT_PADDING
} pool_t;

#pragma pack()



//#define DEBUG_MALLOCS
#ifdef DEBUG_MALLOCS
#define MDEBUG(msg, ptr, length) debug_malloc_printf(msg, ptr, length)
static void debug_malloc_printf(const char* msg, const void* ptr, unsigned length);
#else
#define MDEBUG(...) 
#endif 
static void* _malloc(uint32_t size, uint32_t alignment);
static void _free(void *p);


static pool_t *memory_pool = nullptr;


/*************************************************************************************************/
extern "C" void* heap_malloc(uint32_t size)
{
    return heap_malloc_aligned(size, sizeof(uint32_t)); // always align to a word
}

/*************************************************************************************************/
extern "C" void* heap_malloc_aligned(uint32_t size, uint32_t alignment)
{
    assert(alignment != 0);

    void* ptr = _malloc(size, alignment);

    // TODO: actually align ptr to specified alignement ...
    
    if(ptr != nullptr)
    {
        memset(ptr, 0, size);
    } 
    else
    {
       assert(!"Malloc failed");
    }
    
    return ptr;
}

/*************************************************************************************************/
extern "C" void heap_free(void* ptr)
{
    if(ptr == nullptr)
    {
         assert(!"Attempting to free a null pointer");
    }
    else
    {
        _free(ptr);
    }
}

/*************************************************************************************************/
extern "C" void heap_set_buffer(void* buffer, uint32_t length)
{
    if(length != 0)
    {
        uint8_t* buffer_ptr = reinterpret_cast<uint8_t*>(buffer);
        uintptr_t buffer_addr = (uintptr_t)buffer_ptr;
        uint8_t unaligned_amount = (buffer_addr & (sizeof(uintptr_t)-1));
        uint8_t align_amount = (unaligned_amount > 0) ? sizeof(uintptr_t) - unaligned_amount : 0;
        buffer_ptr += align_amount;
        length -= align_amount;

        auto pool = reinterpret_cast<pool_t*>(buffer_ptr);

        memset(buffer_ptr, 0, length);

        pool->base.size = 0;
        pool->base.ptr = pool->freep = &pool->base;

        auto start = (header_t*)((uint8_t*)pool + sizeof(pool_t));
        start->size = (length - sizeof(pool_t)) / sizeof(header_t);
        pool->total_size = start->size * sizeof(header_t);

        memory_pool = pool;

        _free(&start[1]);
    }
    else
    {
        memory_pool = reinterpret_cast<pool_t*>(buffer);
    }
}

/*************************************************************************************************/
bool heap_get_buffer(void** buffer_ptr)
{
    if(memory_pool == nullptr)
    {
        return false;
    }

    *buffer_ptr = memory_pool;

    return true;
}

/*************************************************************************************************/
bool heap_get_stats(HeapStats *stats)
{
    header_t *p, *prevp;
    uint32_t available_blocks = 0;

    if(memory_pool == nullptr)
    {
        assert(!"Must call heap_set_buffer() first");
        return false;
    }

    acquire_lock();

    prevp = memory_pool->freep;

    for (p = prevp->ptr;; prevp = p, p = p->ptr)
    {
        available_blocks += p->size;

        if (p == memory_pool->freep)
        {
            break;
        }
    }

    release_lock();

    stats->size = memory_pool->total_size;
    stats->remaining = available_blocks * sizeof(header_t);
    stats->used = stats->size - stats->remaining;

    return true;
}

/*************************************************************************************************/
static void* _malloc(uint32_t length, uint32_t alignment)
{
    header_t *p, *prevp;
    void* retval = nullptr;

    unsigned aligned_length = align_up(length, sizeof (header_t));
    assert(alignment <= sizeof (header_t));

    /*
     * Smallest multiple of sizeof (headers), which the
     * can contain required number of bytes, plus 1 for the header itself:
     */
    unsigned num_blocks = aligned_length / sizeof (header_t) + 1;


    if(memory_pool == nullptr)
    {
        assert(!"Must call heap_set_buffer() first");
        return nullptr;
    }

    acquire_lock();

    prevp = memory_pool->freep;

    for (p = prevp->ptr;; prevp = p, p = p->ptr)
    {
        /*
         * P mesh opening the free-list, followed by prevp, no
         * Termination condition!
         */

        if (p->size >= num_blocks)          /* If p is large enough? */
        {
            if (p->size == num_blocks)     /* If exactly, he is ... */
            {
                prevp->ptr = p->ptr ;      /* ... the list */
            }
            else                                /* otherwise ... */
            {
                p->size -= num_blocks;     /* P is reduced */
                p += p->size;                /* ... And the last part */
                p->size = num_blocks;      /* ... The block ... */
            }
            memory_pool->freep = prevp;
            MDEBUG("malloc", &p[1], length);
            retval = (void *) &p[1];              /* return but at the address of p + sizeof(header_t),
                                                   since p points to the header. */
            break;
        }

        if (p == memory_pool->freep)                   /* the list does not contain a block big enough */
        {
            break;
        }
    }

    release_lock();

    if(retval == nullptr)
    {
        MDEBUG("malloc failed", nullptr, length);
    }

    return retval;
}

/*************************************************************************************************/
void _free (void * ap)
{
    header_t *bp, *p;
    int max_scans = 1000000;

    // Ensure the pointer is a valid RAM address
    if(ap == nullptr)
    {
        return;
    }

    acquire_lock();

    bp = (header_t*)(((uint8_t*)ap) - sizeof(header_t)); /* Here is the header of the block */

    // Ensure the header pointer is aligned
    if(((uintptr_t)bp & (sizeof(header_t)-1)) != 0)
    {
        assert(!"Invalid address given to free()");
        goto exit;
    }

    /* The list is scanned, the block is to the Address size are inserted
     * after properly for userspace defragmentation.
     */

    for(p = memory_pool->freep; !(bp> p && bp <p->ptr); p = p->ptr, --max_scans)
    {
        if(p >= p->ptr && (bp > p || bp < p->ptr))
        {
            break; /* Bp is present block with the smallest or behind  Largest block address */
        }
            
        if(max_scans <= 0)
        {
            assert(!"Heap corruption");
            goto exit;
        }
    }

    /* Association with upper neighbors */
    if(bp + bp->size == p->ptr)
    {
        bp->size += p->ptr->size;
        bp->ptr = p->ptr->ptr;
    }
    else
    {
        bp->ptr = p->ptr;
    }

    /* Union with lower neighbor */
    if (p + p->size == bp)
    {
        p->size += bp->size;
        p->ptr = bp->ptr;
    }
    else
    {
        p->ptr = bp;
    }

    memory_pool->freep = p;

    MDEBUG("free", ap, bp->size*sizeof(header_t));

    exit: 
    release_lock();
}


#ifndef __arm__
#include <mutex>

// FIXME: This is causing linking errors when building the python wrapper DLL
//static std::mutex memory_lock;


/*************************************************************************************************/
void acquire_lock()
{
  //  memory_lock.lock();
}

/*************************************************************************************************/
void release_lock()
{
  //  memory_lock.unlock();
}

#endif //  __arm__


#ifdef DEBUG_MALLOCS
#include <stdio.h>
extern "C" int _write(int file, const char *ptr, int len);

/*************************************************************************************************/
static void debug_malloc_printf(const char* msg, const void* ptr, unsigned length)
{
    char buffer[64];
    HeapStats stats;

    heap_get_stats(&stats);

    int l = snprintf(buffer, sizeof(buffer), "%s: %p = %d (heap used: %d)\n", msg, ptr, length, stats.used);
    _write(0, buffer, l);
}
#endif // DEBUG_MALLOCS

#endif // MLTK_DLL_IMPORT
