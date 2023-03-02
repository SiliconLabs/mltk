#include "sl_event_handler.h"

#include "em_chip.h"
#include "sl_device_init_nvic.h"
#include "sl_board_init.h"
#include "sl_device_init_dcdc.h"
#include "sl_device_init_hfrco.h"
#include "sl_hfxo_manager.h"
#include "sl_device_init_hfxo.h"
#include "sl_device_init_lfxo.h"
#include "sl_device_init_clocks.h"
#include "sl_device_init_emu.h"
#include "sl_board_control.h"
#include "sl_sleeptimer.h"
#include "gpiointerrupt.h"
#include "sl_iostream_init_eusart_instances.h"
#include "nvm3_default.h"
#include "sl_simple_button_instances.h"
#include "sl_simple_led_instances.h"
#include "sl_iostream_init_instances.h"
#include "sl_power_manager.h"

#ifdef SL_CATALOG_KERNEL_PRESENT
#include "cmsis_os2.h"
#endif

#ifdef SL_CATALOG_BLUETOOTH_PRESENT
#include "pa_conversions_efr32.h"
#include "sl_rail_util_pti.h"
#include "psa/crypto.h"
#include "sli_protocol_crypto.h"
#include "sl_bluetooth.h"
#include "sl_mbedtls.h"
#include "sl_bt_rtos_adaptation.h"
#endif




void sl_platform_init(void)
{
  CHIP_Init();
  sl_device_init_nvic();
  sl_board_preinit();
  sl_device_init_dcdc();
  sl_hfxo_manager_init_hardware();
  sl_device_init_hfxo();
  sl_device_init_lfxo();
  sl_device_init_dpll();
  sl_device_init_clocks();
  sl_device_init_emu();
  sl_board_init();
#ifdef SL_CATALOG_BLUETOOTH_PRESENT
  nvm3_initDefault();
#endif
#ifdef SL_CATALOG_KERNEL_PRESENT
  osKernelInitialize();
#endif
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
}

void sl_service_init(void)
{
  sl_board_configure_vcom();
  sl_sleeptimer_init();
  sl_hfxo_manager_init();

#ifdef SL_CATALOG_BLUETOOTH_PRESENT
  sl_mbedtls_init();
  psa_crypto_init();
  sli_aes_seed_mask();
#endif

  sl_iostream_init_instances();
}

void sl_stack_init(void)
{
#ifdef SL_CATALOG_BLUETOOTH_PRESENT
  sl_rail_util_pa_init();
  sl_rail_util_pti_init();
  sl_bt_rtos_init();
#endif
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
#ifdef SL_CATALOG_BLUETOOTH_PRESENT
  sl_bt_step();
#endif
}

void sl_internal_app_process_action(void)
{
}

void sl_iostream_init_instances(void)
{
  sl_iostream_eusart_init_instances();
}

