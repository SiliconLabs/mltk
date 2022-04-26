#pragma once 
#include <stdio.h>

#include "sl_sleeptimer.h"
#include "fingerprint_reader_config.h"
#include "fingerprint_reader.h"



#define SL_CONCAT(A, B, C) A ## B ## C
#define USART_CLOCK(N) SL_CONCAT(cmuClock_USART, N, )
#define USART_PORT(N) SL_CONCAT(USART, N, )
#define USART_RX_DMA_SIGNAL(N) SL_CONCAT(dmadrvPeripheralSignal_USART, N, _RXDATAV)

#define FINGERPRINT_READER_USART USART_PORT(FINGERPRINT_READER_USART_PERIPHERAL_NO)


#ifdef FINGERPRINT_READER_DEBUG_ENABLED
#define FPR_DEBUG(msg, ...) printf("[FPR] " msg "\n", ## __VA_ARGS__)
#else 
#define FPR_DEBUG(...)
#endif


#define DELAY_MS(ms) sl_sleeptimer_delay_millisecond(ms)