#include <stdint.h>

#include "em_device.h"


#define PLATFORM_FLASH_START_ADDRESS                (FLASH_BASE)
#define PLATFORM_FLASH_END_ADDRESS                  (PLATFORM_FLASH_START_ADDRESS + FLASH_SIZE)
#define PLATFORM_IS_FLASH_ADDRESS(addr)             ((((uint32_t)addr) >= PLATFORM_FLASH_START_ADDRESS) && (((uint32_t)addr) < PLATFORM_FLASH_END_ADDRESS))



__STATIC_INLINE void _jumpto(uint32_t addr )
{
    addr |= 0x00000001;  /* Last bit of jump address indicates whether destination is Thumb or ARM code */
    __ASM volatile ("BX %0" : : "r" (addr) :);
}



/*************************************************************************************************
 * This is the entry point for the bootloader device memory
 *
 * This simply jumps to the reset handler if a valid address is found
 */
void __NO_RETURN __attribute__((used, section(".bootloader"))) bootloader_entry(void)
{
    const volatile uint32_t *vector_table_base = (const uint32_t*)FLASH_BASE;

    // If address is a valid flash address
    if(PLATFORM_IS_FLASH_ADDRESS(vector_table_base[1]))
    {
        // Configure the main stack pointer
        __set_MSP(vector_table_base[0]);

        // Then jump to the reset handler
        _jumpto(vector_table_base[1]);
    }

    // Otherwise spin forever
    for(;;)
    {
        __asm__("wfe");
    }
}
