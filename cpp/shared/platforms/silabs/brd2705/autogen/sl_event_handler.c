#include "sl_event_handler.h"

#include "em_chip.h"
#include "sl_device_init_nvic.h"
#include "sl_board_init.h"
#include "sl_device_init_dcdc.h"
#include "sl_device_init_lfxo.h"
#include "sl_device_init_hfrco.h"
#include "sl_device_init_hfxo.h"
#include "sl_device_init_clocks.h"
#include "sl_device_init_emu.h"
#include "sl_board_control.h"
#include "sl_sleeptimer.h"
#include "gpiointerrupt.h"
#include "sl_iostream_init_eusart_instances.h"
#include "sl_simple_button_instances.h"
#include "sl_simple_led_instances.h"
//#include "sl_tflite_micro_init.h"
#include "sl_iostream_init_instances.h"
#include "sl_power_manager.h"
#include "sl_cos.h"


#ifdef SL_CATALOG_KERNEL_PRESENT
#include "cmsis_os2.h"
#endif


void sl_platform_init(void)
{
  CHIP_Init();
  sl_device_init_nvic();
  sl_board_preinit();
  sl_device_init_dcdc();
  sl_device_init_lfxo();
  sl_device_init_hfrco();
  sl_device_init_clocks();
  sl_device_init_emu();
  sl_board_init();
  sl_power_manager_init();
}

void sl_kernel_start(void)
{
#ifdef SL_CATALOG_KERNEL_PRESENT
  osKernelStart();
#endif
}



void sl_driver_init(void)
{
  GPIOINT_Init();
  sl_simple_button_init_instances();
  sl_simple_led_init_instances();
  sl_cos_send_config();
}

void sl_service_init(void)
{
  sl_board_configure_vcom();
  sl_sleeptimer_init();
  sl_iostream_init_instances();
}

void sl_stack_init(void)
{
}

void sl_internal_app_init(void)
{
}

void sl_platform_process_action(void)
{
}

void sl_service_process_action(void)
{
}

void sl_stack_process_action(void)
{
}

void sl_internal_app_process_action(void)
{
}

void sl_iostream_init_instances(void)
{
  sl_iostream_eusart_init_instances();
}

void sli_sleeptimer_set_pm_em_requirement() {}