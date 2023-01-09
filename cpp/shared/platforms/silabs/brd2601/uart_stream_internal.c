#include "src/uart_stream_internal.h"

#include "sl_iostream_handles.h"
#include "sl_iostream_eusart.h"
#include "sl_iostream_eusart_vcom_config.h"


#define SL_CONCAT(A, B, C) A ## B ## C
/* Generate the cmu clock symbol based on instance. */
#define RX_DMA_SIGNAL(N) SL_CONCAT(ldmaPeripheralSignal_EUSART, N, _RXFL)
#define TX_DMA_SIGNAL(N) SL_CONCAT(ldmaPeripheralSignal_EUSART, N, _TXFL)


static EUSART_TypeDef *peripheral;
uint8_t uart_stream_initialized = 0;


/*************************************************************************************************/
bool uart_stream_internal_init(uint32_t baud_rate)
{
    sl_iostream_t* stream = sl_iostream_get_handle("vcom");
    if(stream == NULL)
    {
        return false;
    }

    sl_iostream_eusart_context_t *context = stream->context;
    peripheral = context->eusart;

    if(baud_rate != 0)
    {
        EUSART_BaudrateSet(peripheral, 0, baud_rate);
    }


    uart_stream_initialized = 1;

    return true;
}

/*************************************************************************************************/
void uart_stream_internal_get_dma_config(UartStreamDmaConfig *config)
{
    const LDMA_TransferCfg_t cfg = LDMA_TRANSFER_CFG_PERIPHERAL(0);
    config->rx_cfg = cfg;
    config->rx_cfg.ldmaReqSel = RX_DMA_SIGNAL(SL_IOSTREAM_EUSART_VCOM_PERIPHERAL_NO);
    config->rx_address = (uint32_t)&peripheral->RXDATA;

    config->tx_cfg = cfg;
    config->tx_cfg.ldmaReqSel = TX_DMA_SIGNAL(SL_IOSTREAM_EUSART_VCOM_PERIPHERAL_NO);
    config->tx_address = (uint32_t)&peripheral->TXDATA;
}

__attribute__((weak)) void uart_stream_internal_rx_irq_callback(){}

/*************************************************************************************************/
void uart_stream_internal_irq_handler()
{
    EUSART_IntClear(peripheral, EUSART_IF_RXFL);
    uart_stream_internal_rx_irq_callback();
}

/*************************************************************************************************/
uint32_t uart_stream_internal_set_irq_enabled(bool enable)
{
    uint32_t current_state = EUSART_IntGetEnabled(peripheral);

    if(enable)
    {
        EUSART_IntEnable(peripheral, EUSART_IF_RXFL);
    }
    else
    {
        EUSART_IntDisable(peripheral, EUSART_IF_RXFL);
    }

    return current_state;
}

/*************************************************************************************************/
void uart_stream_internal_write_data(const void* data, uint32_t length)
{
    const uint8_t* ptr = data;
    for(; length > 0; --length, ++ptr)
    {
        EUSART_Tx(peripheral, *ptr);
    }
}



/*************************************************************************************************/
int32_t uart_stream_internal_read_char()
{
    if (!(peripheral->STATUS & EUSART_STATUS_RXFL))
    {
        return -1;
    }

    return (uint8_t)peripheral->RXDATA;
}