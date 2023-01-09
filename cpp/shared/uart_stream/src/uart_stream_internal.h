#pragma once


#include <stdint.h>

#include "em_ldma.h"


#ifdef __cplusplus
extern "C" {
#endif


#define UART_STREAM_INVOKE_IRQ() if(uart_stream_initialized) { uart_stream_internal_irq_handler(); return; }

extern uint8_t uart_stream_initialized;


typedef struct UartStreamDmaConfig
{
    LDMA_TransferCfg_t rx_cfg;
    uint32_t rx_address;
    LDMA_TransferCfg_t tx_cfg;
    uint32_t tx_address;
} UartStreamDmaConfig;


bool uart_stream_internal_init(uint32_t baud_rate);
void uart_stream_internal_irq_handler();
void uart_stream_internal_rx_irq_callback();
uint32_t uart_stream_internal_set_irq_enabled(bool enabled);
void uart_stream_internal_write_data(const void* data, uint32_t length);
void uart_stream_internal_get_dma_config(UartStreamDmaConfig *config);
int32_t uart_stream_internal_read_char();



#ifdef __cplusplus
}
#endif