
#include <stdio.h>


extern int _write(int file, const char *ptr, int len);


void HardFault_Handler   (void)
{
#define msg "\n\nHardfault triggered!\n\n"
    _write(0, msg, sizeof(msg)-1);

    __asm( "MRS R0, MSP" );
    __asm( "MRS R1, PSP" );
    __asm( "MOV R2, LR" );


   __asm( "bkpt" );

    while ( 1 )
    {
        __asm( "wfe" );
    }
}

