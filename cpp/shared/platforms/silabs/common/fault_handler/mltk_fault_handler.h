#pragma once
#include "em_device.h"




#define MLTK_ADD_FAULT_HANDLER() \
{ \
    __ASM volatile ("MRS R0, MSP" ); \
    __ASM volatile ("MRS R1, PSP" ); \
    __ASM volatile ("MOV R2, LR" ); \
    __ASM volatile ("B mltk_fault_handler"); \
}

extern void __NO_RETURN mltk_fault_handler(uint32_t MSP, uint32_t PSP, uint32_t LR);

