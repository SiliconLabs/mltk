#include <stdint.h>
#include <stdlib.h>


extern const uint32_t __init_array_start;
extern const uint32_t __init_array_end;
extern const uint8_t __HeapBase;
extern const uint8_t __HeapLimit;


extern int main(void);
extern void heap_set_buffer(void* buffer, uint32_t length);


static char mltk_stack[MLTK_STACK_SIZE] __attribute__ ((aligned(8), used, section(".stack")));


/*************************************************************************************************/
void _start(void)
{
    // Configure the MLTK malloc memory util
    heap_set_buffer((uint8_t*)&__HeapBase, (uint32_t)(&__HeapLimit - &__HeapBase));

    // Initialize any global static C++ constructors
    for(const uint32_t *p = &__init_array_start; p < &__init_array_end; ++p)
    {
        void (**constructor_ptr)() = (void*)p;

        (*constructor_ptr)();
    }

    main();

    for(;;)
    {
        __asm__("wfe");
    }
}