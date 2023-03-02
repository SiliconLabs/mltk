#include <stdio.h>
#include "em_device.h"
#include "mltk_fault_handler.h"


#pragma GCC optimize ("O0")



#define SCB_AIRCR_VECTKEY                   ( (unsigned long)( 0x5FA << SCB_AIRCR_VECTKEY_Pos ))

#define  SCB_CFSR_IACCVIOL                   ((uint32_t)0x00000001)
#define  SCB_CFSR_DACCVIOL                   ((uint32_t)0x00000002)
#define  SCB_CFSR_MUNSTKERR                  ((uint32_t)0x00000008)
#define  SCB_CFSR_MSTKERR                    ((uint32_t)0x00000010)
#define  SCB_CFSR_MMARVALID                  ((uint32_t)0x00000080)
#define  SCB_CFSR_IBUSERR                    ((uint32_t)0x00000100)
#define  SCB_CFSR_PRECISERR                  ((uint32_t)0x00000200)
#define  SCB_CFSR_IMPRECISERR                ((uint32_t)0x00000400)
#define  SCB_CFSR_UNSTKERR                   ((uint32_t)0x00000800)
#define  SCB_CFSR_STKERR                     ((uint32_t)0x00001000)
#define  SCB_CFSR_BFARVALID                  ((uint32_t)0x00008000)
#define  SCB_CFSR_UNDEFINSTR                 ((uint32_t)0x00010000)
#define  SCB_CFSR_INVSTATE                   ((uint32_t)0x00020000)
#define  SCB_CFSR_INVPC                      ((uint32_t)0x00040000)
#define  SCB_CFSR_NOCP                       ((uint32_t)0x00080000)
#define  SCB_CFSR_UNALIGNED                  ((uint32_t)0x01000000)
#define  SCB_CFSR_DIVBYZERO                  ((uint32_t)0x02000000)



typedef enum
{
    HANDLER_MSP_MSP = 0xF1, /* Return to Handler mode. Exception return gets state from MSP. Execution uses MSP after return. */
    THREAD_MSP_MSP  = 0xF9, /* Return to Thread mode.  Exception return gets state from MSP. Execution uses MSP after return. */
    THREAD_PSP_PSP  = 0xFD  /* Return to Thread mode.  Exception return gets state from PSP. Execution uses PSP after return. */
} exc_return_t;

typedef struct
{
    /* Stacked registers */
    uint32_t R0;
    uint32_t R1;
    uint32_t R2;
    uint32_t R3;
    uint32_t R12;
    uint32_t LR;
    uint32_t PC;  /* (Return Address) */
    uint32_t PSR;
} exception_context_t;




#define PRINT_STR(x) _asset_write(x "\n")
#define PRINTF(fmt, ...) \
{ \
    char buffer[256]; \
    snprintf(buffer, sizeof(buffer), fmt, ## __VA_ARGS__); \
    _asset_write(buffer); \
}
#define DUMP_REG(name, val) PRINTF("%-10s = 0x%08X\n", name, val)
extern void _asset_write(const char *s);

#ifdef DEBUG
#define DEBUG_BREAKPOINT() __BKPT()
#else
#define DEBUG_BREAKPOINT()
#endif


__STATIC_FORCEINLINE uint32_t __get_LR(void)
{
  uint32_t result;

  __ASM volatile ("MOV %0, LR" : "=r" (result) );
  return(result);
}


void __NO_RETURN mltk_fault_handler(uint32_t MSP, uint32_t PSP, uint32_t LR)
{
    exception_context_t*  stackframe = NULL;
    const volatile uint32_t CTRL = __get_CONTROL();
    const volatile uint32_t IPSR = __get_IPSR();
    const volatile uint32_t APSR = __get_APSR();
    const volatile uint32_t xPSR = __get_xPSR();
    const volatile uint32_t PRIMASK = __get_PRIMASK();
    const volatile uint32_t BASEPRI = __get_BASEPRI();
    const volatile uint32_t FAULTMSK = __get_FAULTMASK();
    const volatile uint32_t MMFAR = SCB->MMFAR;
    const volatile uint32_t BFAR = SCB->BFAR;
    const int32_t irq_number = (int32_t)((SCB->ICSR >> SCB_ICSR_VECTACTIVE_Pos) & SCB_ICSR_VECTACTIVE_Msk) - 16;

    __set_PRIMASK( 0x01 );

    /* Get the Link Register value which contains the EXC_RETURN code */
    const exc_return_t exc_return = LR & 0xff;

    /* The location of the stack frame of the offending code is indicated by the EXC_RETURN code */
    if ( ( exc_return & 0x00000004 ) != 0 )
    {
        stackframe = (exception_context_t*) PSP;
    }
    else
    {
        stackframe = (exception_context_t*) MSP;
    }

    if(irq_number < 0)
    {
        #define _stringify_(x) #x
        #define _stringify(x) _stringify_(x)
        #define _add_mapping(name) [name ## _IRQn + 16] = _stringify(name)
        #define _add_unknown_mapping(number) [number + 16] = "Unknown"
        static const char* const IRQ_MAP[16] =
        {
            _add_unknown_mapping(-16),
            _add_unknown_mapping(-15),
            _add_mapping(NonMaskableInt),
            _add_mapping(HardFault),
            _add_mapping(MemoryManagement),
            _add_mapping(BusFault),
            _add_mapping(UsageFault),
            _add_unknown_mapping(-9),
            _add_unknown_mapping(-8),
            _add_unknown_mapping(-7),
            _add_unknown_mapping(-6),
            _add_mapping(SVCall),
            _add_mapping(DebugMonitor),
            _add_unknown_mapping(-3),
            _add_mapping(PendSV),
            _add_mapping(SysTick),
        };
        PRINTF("\n\nCortex IRQ %s triggered!\n\n", IRQ_MAP[irq_number + 16]);
    }
    else
    {
        PRINTF("\n\nUnhandled IRQ number=%d triggered!\n\n", irq_number);
    }


    PRINT_STR("Registers dump:");
    DUMP_REG("PC", stackframe->PC);
    DUMP_REG("LR", stackframe->LR);
    DUMP_REG("CTRL", CTRL);
    DUMP_REG("IPSR", IPSR);
    DUMP_REG("APSR", APSR);
    DUMP_REG("xPSR", xPSR);
    DUMP_REG("PRIMASK", PRIMASK);
    DUMP_REG("BASEPRI", BASEPRI);
    DUMP_REG("FAULTMSK", FAULTMSK);
    DUMP_REG("MSP",  MSP);
    DUMP_REG("PSP",  PSP);
    DUMP_REG("SCB->MMFAR", SCB->MMFAR);
    DUMP_REG("SCB->BFAR", SCB->BFAR);
    DUMP_REG("SCB->HFSR", SCB->HFSR);
    DUMP_REG("SCB->CFSR", SCB->CFSR);
    DUMP_REG("SCB->ICSR", SCB->ICSR);
    DUMP_REG("SCB->AFSR", SCB->AFSR);
    PRINT_STR("\n");

    /* Find cause of hardfault */
    if ( ( SCB->HFSR & SCB_HFSR_VECTTBL_Msk ) != 0 )
    {
        PRINT_STR("Details: Vector Table Hard Fault - Bus fault during vector table read during exception processing.");
        __BKPT(); /* Vector Table Hard Fault - Bus fault during vector table read during exception processing. */
    }
    else if ( ( SCB->HFSR & SCB_HFSR_FORCED_Msk ) != 0 )
    {
        /* Hard Fault is an escalated fault that was not handled */
        /* Need to read the other fault status registers */

        if ( ( SCB->CFSR & SCB_CFSR_IACCVIOL ) != 0 )
        {
            /* Memory Management Fault */
            PRINT_STR("Details: Instruction Access Violation - Attempt to execute an instruction from a region marked Execute Never");
            __BKPT();  /* Instruction Access Violation - Attempt to execute an instruction from a region marked Execute Never */
            (void) stackframe->LR; /* Check this variable for the jump instruction that jumped to an invalid region */
            (void) stackframe->PC; /* Check this variable for the location that was attempted to be executed */
                                   /* You may try stepping past the return of this handler, which may return near the location of the error */

        }
        else if ( ( SCB->CFSR & SCB_CFSR_DACCVIOL     ) != 0 )
        {
            /* Memory Management Fault */
            PRINT_STR("Details: Data Access Violation");
            DEBUG_BREAKPOINT();  /* Data Access Violation */
            (void) stackframe->PC; /* Check this variable for the location of the offending instruction */
            (void) MMFAR;           /* Check this variable for the address of the attempted access */
            /* You may try stepping past the return of this handler, which may return near the location of the error */
        }
        else if ( ( SCB->CFSR & SCB_CFSR_MUNSTKERR    ) != 0 )
        {
            /* Memory Management Fault */
            PRINT_STR("Details: Unstacking fault returning from an exception - stack possibly corrupted during exception handler");
            DEBUG_BREAKPOINT();  /* Unstacking fault returning from an exception - stack possibly corrupted during exception handler */
                                   /* New stackframe is not saved in this case */
        }
        else if ( ( SCB->CFSR & SCB_CFSR_MSTKERR      ) != 0 )
        {
            /* Memory Management Fault */
            PRINT_STR("Details: Stacking fault whilst entering an exception - probably a bad stack pointer");
            __BKPT();  /* Stacking fault whilst entering an exception - probably a bad stack pointer */
                                   /* Stack frame may be incorrect due to bad stack pointer */

        }
        else if ( ( SCB->CFSR & SCB_CFSR_IBUSERR      ) != 0 )
        {
            /* Bus Fault */
            PRINT_STR("Details: Instruction Bus Error whilst fetching an instruction");
            DEBUG_BREAKPOINT();  /* Instruction Bus Error whilst fetching an instruction*/
        }
        else if ( ( SCB->CFSR & SCB_CFSR_PRECISERR    ) != 0 )
        {
            /* Bus Fault */
            PRINT_STR("Details: Precise Data Bus Error - i.e. Data Bus fault at well defined location");
            DEBUG_BREAKPOINT();  /* Precise Data Bus Error - i.e. Data Bus fault at well defined location */
            (void) stackframe->PC; /* Check this variable for the location of the offending instruction */
            (void) BFAR;           /* Check this variable for the faulting address */
                                   /* You may try stepping past the return of this handler, which may return near the location of the error */
        }
        else if ( ( SCB->CFSR & SCB_CFSR_IMPRECISERR  ) != 0 )
        {
            /* Bus Fault */
            PRINT_STR("Details: Imprecise Data Bus Error - i.e. Data Bus fault occurred but details have been lost due to priorities delaying processing of the fault");
            __BKPT();  /* Imprecise Data Bus Error - i.e. Data Bus fault occurred but details have been lost due to priorities delaying processing of the fault */
                                   /* No fault details are available in this case*/
                                   /* You may try stepping past the return of this handler, which may return near the location of the error */
        }
        else if ( ( SCB->CFSR & SCB_CFSR_UNSTKERR     ) != 0 )
        {
            /* Bus Fault */
            PRINT_STR("Details: Unstacking fault returning from an exception - stack possibly corrupted during exception handler");
            DEBUG_BREAKPOINT();  /* Unstacking fault returning from an exception - stack possibly corrupted during exception handler */
                                   /* New stackframe is not saved in this case */
        }
        else if ( ( SCB->CFSR & SCB_CFSR_STKERR       ) != 0 )
        {
            /* Bus Fault */
            PRINT_STR("Details: Stacking fault whilst entering an exception - probably a bad stack pointer");
            DEBUG_BREAKPOINT();  /* Stacking fault whilst entering an exception - probably a bad stack pointer */
                                   /* Stack frame may be incorrect due to bad stack pointer */

        }
        else if ( ( SCB->CFSR & SCB_CFSR_UNDEFINSTR   ) != 0 )
        {
            /* Usage Fault */
            PRINT_STR("Details: Undefined Instruction Usage fault - probably corrupted memory in code space");
            DEBUG_BREAKPOINT();  /* Undefined Instruction Usage fault - probably corrupted memory in code space */
            (void) stackframe->PC; /* Check this variable for the location of the offending instruction */
                                   /* You may try stepping past the return of this handler, which may return near the location of the error */
        }
        else if ( ( SCB->CFSR & SCB_CFSR_INVSTATE     ) != 0 )
        {
            /* Usage Fault */
            PRINT_STR("Details: Invalid State usage fault - Illegal use of EPSR was attempted");
            DEBUG_BREAKPOINT();  /* Invalid State usage fault - Illegal use of EPSR was attempted */
            (void) stackframe->PC; /* Check this variable for the location of the offending instruction */
                                   /* You may try stepping past the return of this handler, which may return near the location of the error */
        }
        else if ( ( SCB->CFSR & SCB_CFSR_INVPC        ) != 0 )
        {
            /* Usage Fault */
            PRINT_STR("Details: Invalid PC load usage fault - the EXC_RETURN value in LR was invalid on return from an exception - possibly stack corruption in exception");
            DEBUG_BREAKPOINT();  /* Invalid PC load usage fault - the EXC_RETURN value in LR was invalid on return from an exception - possibly stack corruption in exception */
            (void) stackframe->PC; /* Check this variable for the location of the offending instruction */
                                   /* You may try stepping past the return of this handler, which may return near the location of the error */
        }
        else if ( ( SCB->CFSR & SCB_CFSR_NOCP         ) != 0 )
        {
            /* Usage Fault */
            PRINT_STR("Details: No Coprocessor usage fault - coprocessor instruction attempted on processor without support for them");
            DEBUG_BREAKPOINT();  /* No Coprocessor usage fault - coprocessor instruction attempted on processor without support for them */
            (void) stackframe->PC; /* Check this variable for the location of the offending instruction */
                                   /* You may try stepping past the return of this handler, which may return near the location of the error */
        }
        else if ( ( SCB->CFSR & SCB_CFSR_UNALIGNED   ) != 0 )
        {
            /* Usage Fault */
            PRINT_STR("Details: Unaligned access usage fault - Unaligned access whilst UNALIGN_TRP bit of SCB_CCR is set, or any unaligned access to LDM, STM, LDRD or STRD");
            DEBUG_BREAKPOINT();  /* Unaligned access usage fault - Unaligned access whilst UNALIGN_TRP bit of SCB_CCR is set, or any unaligned access to LDM, STM, LDRD or STRD */
            (void) stackframe->PC; /* Check this variable for the location of the offending instruction */
                                   /* You may try stepping past the return of this handler, which may return near the location of the error */
        }
        else if ( ( SCB->CFSR & SCB_CFSR_DIVBYZERO    ) != 0 )
        {
            /* Usage Fault */
            PRINT_STR("Details: Divide by zero usage fault");
            DEBUG_BREAKPOINT();  /* Divide by zero usage fault */
            (void) stackframe->PC; /* Check this variable for the location of the offending instruction */
                                   /* You may try stepping past the return of this handler, which may return near the location of the error */
        }
        else
        {
            /* Unknown Fault */
            PRINT_STR("Details: Unknown Fault");
            DEBUG_BREAKPOINT();
            /* You may try stepping past the return of this handler, which may return near the location of the error */
        }
    }
    else
    {
        /* Unknown Hard Fault cause (possbily watchdog, system monitor, or software exception) */
        PRINT_STR("Details: Unknown Hard Fault cause (possbily watchdog, system monitor, or software exception)");

    }

    PRINT_STR(
        "\nHint: The PC register should contain the address where the fault occurred.\n"
        "See the .map file in the build directory to map this address to the source code.\n"
        "You can also build with the MLTK_ENABLE_OUTPUT_DISASSEMBLY CMake variable to generate a full disassembly dump of the firmware."
    );

    PRINT_STR("\n*** Fault triggered\n");

    __BKPT();

    while ( 1 )
    {
        __WFE();
    }
}


__WEAK void NMI_Handler(void)
{
    MLTK_ADD_FAULT_HANDLER()
}

__WEAK void HardFault_Handler(void)
{
    MLTK_ADD_FAULT_HANDLER()
}

__WEAK void MemManage_Handler(void)
{
    MLTK_ADD_FAULT_HANDLER()
}
__WEAK void BusFault_Handler(void)
{
    MLTK_ADD_FAULT_HANDLER()
}
__WEAK void UsageFault_Handler(void)
{
    MLTK_ADD_FAULT_HANDLER()
}
__WEAK void SMU_NS_PRIVILEGED_IRQHandler(void)
{
    MLTK_ADD_FAULT_HANDLER()
}
__WEAK void SMU_S_PRIVILEGED_IRQHandler(void)
{
    MLTK_ADD_FAULT_HANDLER()
}
__WEAK void SMU_SECURE_IRQHandler(void)
{
    MLTK_ADD_FAULT_HANDLER()
}