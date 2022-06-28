
#include <stdio.h>


extern void _asset_write(const char *s);


void HardFault_Handler   (void)
{
#define msg "\n\nHardfault triggered!\n\n"
    _asset_write(msg);

    __asm( "MRS R0, MSP" );
    __asm( "MRS R1, PSP" );
    __asm( "MOV R2, LR" );


   __asm( "bkpt" );

    while ( 1 )
    {
        __asm( "wfe" );
    }
}

