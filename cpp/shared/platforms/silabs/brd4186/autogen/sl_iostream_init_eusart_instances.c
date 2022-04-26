#include "em_device.h"
#include "sl_iostream.h"
#include "sl_iostream_uart.h"
#include "sl_iostream_eusart.h"
// Include instance config 
 #include "sl_iostream_eusart_vcom_config.h"

// MACROs for generating name and IRQ handler function  
#define SL_IOSTREAM_EUSART_CONCAT_PASTER(first, second, third)        first ##  second ## third
#if defined(EUART_COUNT) && (EUART_COUNT > 0)
#define SL_IOSTREAM_EUSART_TX_IRQ_NUMBER(periph_nbr)     SL_IOSTREAM_EUSART_CONCAT_PASTER(EUART, periph_nbr, _TX_IRQn)   
#define SL_IOSTREAM_EUSART_RX_IRQ_NUMBER(periph_nbr)     SL_IOSTREAM_EUSART_CONCAT_PASTER(EUART, periph_nbr, _RX_IRQn)   
#define SL_IOSTREAM_EUSART_TX_IRQ_HANDLER(periph_nbr)    SL_IOSTREAM_EUSART_CONCAT_PASTER(EUART, periph_nbr, _TX_IRQHandler)  
#define SL_IOSTREAM_EUSART_RX_IRQ_HANDLER(periph_nbr)    SL_IOSTREAM_EUSART_CONCAT_PASTER(EUART, periph_nbr, _RX_IRQHandler)  
#else
#define SL_IOSTREAM_EUSART_TX_IRQ_NUMBER(periph_nbr)     SL_IOSTREAM_EUSART_CONCAT_PASTER(EUSART, periph_nbr, _TX_IRQn)   
#define SL_IOSTREAM_EUSART_RX_IRQ_NUMBER(periph_nbr)     SL_IOSTREAM_EUSART_CONCAT_PASTER(EUSART, periph_nbr, _RX_IRQn)   
#define SL_IOSTREAM_EUSART_TX_IRQ_HANDLER(periph_nbr)    SL_IOSTREAM_EUSART_CONCAT_PASTER(EUSART, periph_nbr, _TX_IRQHandler)  
#define SL_IOSTREAM_EUSART_RX_IRQ_HANDLER(periph_nbr)    SL_IOSTREAM_EUSART_CONCAT_PASTER(EUSART, periph_nbr, _RX_IRQHandler)  
#endif

#if defined(EUART_COUNT) && (EUART_COUNT > 0)
#define SL_IOSTREAM_EUSART_CLOCK_REF(periph_nbr)         SL_IOSTREAM_EUSART_CONCAT_PASTER(cmuClock_, EUART, periph_nbr)  
#else
#define SL_IOSTREAM_EUSART_CLOCK_REF(periph_nbr)         SL_IOSTREAM_EUSART_CONCAT_PASTER(cmuClock_, EUSART, periph_nbr)  
#endif


sl_status_t sl_iostream_eusart_init_vcom(void);


// Instance(s) handle and context variable 
static sl_iostream_uart_t sl_iostream_vcom;
sl_iostream_t *sl_iostream_vcom_handle = &sl_iostream_vcom.stream;
sl_iostream_uart_t *sl_iostream_uart_vcom_handle = &sl_iostream_vcom;
static sl_iostream_eusart_context_t  context_vcom;
static uint8_t  rx_buffer_vcom[SL_IOSTREAM_EUSART_VCOM_RX_BUFFER_SIZE];
sl_iostream_instance_info_t sl_iostream_instance_vcom_info = {
  .handle = &sl_iostream_vcom.stream,
  .name = "vcom",
  .type = SL_IOSTREAM_TYPE_UART,
  .periph_id = SL_IOSTREAM_EUSART_VCOM_PERIPHERAL_NO,
  .init = sl_iostream_eusart_init_vcom,
};



sl_status_t sl_iostream_eusart_init_vcom(void)
{
  sl_status_t status;
#if (SL_IOSTREAM_EUSART_VCOM_ENABLE_HIGH_FREQUENCY == 0)
  EUSART_UartInit_TypeDef init_vcom = EUSART_UART_INIT_DEFAULT_LF;
#else
  EUSART_UartInit_TypeDef init_vcom = EUSART_UART_INIT_DEFAULT_HF;
#endif
  init_vcom.baudrate = SL_IOSTREAM_EUSART_VCOM_BAUDRATE;
  init_vcom.parity = SL_IOSTREAM_EUSART_VCOM_PARITY;
  init_vcom.stopbits = SL_IOSTREAM_EUSART_VCOM_STOP_BITS;

  sl_iostream_eusart_config_t config_vcom = { 
    .eusart = SL_IOSTREAM_EUSART_VCOM_PERIPHERAL,
#if (SL_IOSTREAM_EUSART_VCOM_FLOW_CONTROL_TYPE != uartFlowControlSoftware)
    .flow_control = SL_IOSTREAM_EUSART_VCOM_FLOW_CONTROL_TYPE,
#else
    .flow_control = usartHwFlowControlNone,
#endif
    .enable_high_frequency = SL_IOSTREAM_EUSART_VCOM_ENABLE_HIGH_FREQUENCY,
    .clock = SL_IOSTREAM_EUSART_CLOCK_REF(SL_IOSTREAM_EUSART_VCOM_PERIPHERAL_NO),
#if defined(EUSART_COUNT) && (EUSART_COUNT > 1)
    .port_index = SL_IOSTREAM_EUSART_VCOM_PERIPHERAL_NO,
#endif
    .tx_port = SL_IOSTREAM_EUSART_VCOM_TX_PORT,
    .tx_pin = SL_IOSTREAM_EUSART_VCOM_TX_PIN,
    .rx_port = SL_IOSTREAM_EUSART_VCOM_RX_PORT,
    .rx_pin = SL_IOSTREAM_EUSART_VCOM_RX_PIN,
#if defined(SL_IOSTREAM_EUSART_VCOM_CTS_PORT)
    .cts_port = SL_IOSTREAM_EUSART_VCOM_CTS_PORT,
    .cts_pin = SL_IOSTREAM_EUSART_VCOM_CTS_PIN,
#endif
#if defined(SL_IOSTREAM_EUSART_VCOM_RTS_PORT)
    .rts_port = SL_IOSTREAM_EUSART_VCOM_RTS_PORT,
    .rts_pin = SL_IOSTREAM_EUSART_VCOM_RTS_PIN,
#endif
  };
  sl_iostream_uart_config_t uart_config_vcom = {
    .tx_irq_number = SL_IOSTREAM_EUSART_TX_IRQ_NUMBER(SL_IOSTREAM_EUSART_VCOM_PERIPHERAL_NO),
    .rx_irq_number = SL_IOSTREAM_EUSART_RX_IRQ_NUMBER(SL_IOSTREAM_EUSART_VCOM_PERIPHERAL_NO),
    .rx_buffer = rx_buffer_vcom,
    .rx_buffer_length = SL_IOSTREAM_EUSART_VCOM_RX_BUFFER_SIZE,
    .lf_to_crlf = SL_IOSTREAM_EUSART_VCOM_CONVERT_BY_DEFAULT_LF_TO_CRLF,
    .rx_when_sleeping = SL_IOSTREAM_EUSART_VCOM_RESTRICT_ENERGY_MODE_TO_ALLOW_RECEPTION,
#if (SL_IOSTREAM_EUSART_VCOM_FLOW_CONTROL_TYPE == uartFlowControlSoftware)
    .sw_flow_control = true,
#else
    .sw_flow_control = false,
#endif
  };

  // Instantiate eusart instance 
  status = sl_iostream_eusart_init(&sl_iostream_vcom,
                                  &uart_config_vcom,
                                  &init_vcom,
                                  &config_vcom,
                                  &context_vcom);
  EFM_ASSERT(status == SL_STATUS_OK);

  return status;
}



void sl_iostream_eusart_init_instances(void)
{
  // Instantiate eusart instance(s) 
  
  sl_iostream_eusart_init_vcom();
  
}


void SL_IOSTREAM_EUSART_TX_IRQ_HANDLER(SL_IOSTREAM_EUSART_VCOM_PERIPHERAL_NO)(void)
{
  sl_iostream_eusart_irq_handler(sl_iostream_vcom.stream.context);
}

void SL_IOSTREAM_EUSART_RX_IRQ_HANDLER(SL_IOSTREAM_EUSART_VCOM_PERIPHERAL_NO)(void)
{
  sl_iostream_eusart_irq_handler(sl_iostream_vcom.stream.context);
}

#if defined(SL_CATALOG_POWER_MANAGER_PRESENT) && !defined(SL_CATALOG_KERNEL_PRESENT)
 
sl_power_manager_on_isr_exit_t sl_iostream_eusart_vcom_sleep_on_isr_exit(void)
{
  return sl_iostream_uart_sleep_on_isr_exit(&sl_iostream_vcom);
}

#endif
