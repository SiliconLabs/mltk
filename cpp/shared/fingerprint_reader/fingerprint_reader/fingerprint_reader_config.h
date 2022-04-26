
// Fingerprint reader peripheral configuration

// Just use the standard SPI expansion header pinout by default
// NOTE: Technically, the reader uses a UART interface,
///      but we use SPI so that we can repurpose the CS pin

#include "sl_spidrv_exp_config.h"

#ifndef SL_SPIDRV_EXP_TX_LOC
#define SL_SPIDRV_EXP_TX_LOC 0
#endif
#ifndef SL_SPIDRV_EXP_RX_LOC
#define SL_SPIDRV_EXP_RX_LOC 0
#endif

#define FINGERPRINT_READER_USART_PERIPHERAL                SL_SPIDRV_EXP_PERIPHERAL
#define FINGERPRINT_READER_USART_PERIPHERAL_NO             SL_SPIDRV_EXP_PERIPHERAL_NO


#define FINGERPRINT_READER_USART_TX_PORT                   SL_SPIDRV_EXP_TX_PORT        
#define FINGERPRINT_READER_USART_TX_PIN                    SL_SPIDRV_EXP_TX_PIN
#define FINGERPRINT_READER_USART_TX_LOC                    SL_SPIDRV_EXP_TX_LOC

#define FINGERPRINT_READER_USART_RX_PORT                   SL_SPIDRV_EXP_RX_PORT      
#define FINGERPRINT_READER_USART_RX_PIN                    SL_SPIDRV_EXP_RX_PIN
#define FINGERPRINT_READER_USART_RX_LOC                    SL_SPIDRV_EXP_RX_LOC


#define FINGERPRINT_READER_ACTIVITY_PORT                   SL_SPIDRV_EXP_CS_PORT        
#define FINGERPRINT_READER_ACTIVITY_PIN                    SL_SPIDRV_EXP_CS_PIN



#define FINGERPRINT_READER_DUMP_SENSOR_INFO
//#define FINGERPRINT_READER_DEBUG_ENABLED