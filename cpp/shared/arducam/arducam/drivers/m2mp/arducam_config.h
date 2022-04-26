/**
 * Arducam peripheral configuration
 * 
 */


// Just use standard the I2C expansion header pinout by default

#include "sl_i2cspm_sensor_config.h"

#ifndef SL_I2CSPM_SENSOR_SCL_LOC
#define SL_I2CSPM_SENSOR_SCL_LOC 0 
#endif
#ifndef SL_I2CSPM_SENSOR_SDA_LOC
#define SL_I2CSPM_SENSOR_SDA_LOC 0 
#endif


#define ARDUCAM_I2C_PERIPHERAL                  SL_I2CSPM_SENSOR_PERIPHERAL
#define ARDUCAM_I2C_PERIPHERAL_NO               SL_I2CSPM_SENSOR_PERIPHERAL_NO


#define ARDUCAM_I2C_SDA_PORT                    SL_I2CSPM_SENSOR_SDA_PORT
#define ARDUCAM_I2C_SDA_PIN                     SL_I2CSPM_SENSOR_SDA_PIN
#define ARDUCAM_I2C_SDA_LOC                     SL_I2CSPM_SENSOR_SDA_LOC

#define ARDUCAM_I2C_SCL_PORT                    SL_I2CSPM_SENSOR_SCL_PORT
#define ARDUCAM_I2C_SCL_PIN                     SL_I2CSPM_SENSOR_SCL_PIN
#define ARDUCAM_I2C_SCL_LOC                     SL_I2CSPM_SENSOR_SCL_LOC



// Just use the standard SPI expansion header pinout by default

#include "sl_spidrv_exp_config.h"


#ifndef SL_SPIDRV_EXP_TX_LOC
#define SL_SPIDRV_EXP_TX_LOC 0
#endif
#ifndef SL_SPIDRV_EXP_RX_LOC
#define SL_SPIDRV_EXP_RX_LOC 0
#endif
#ifndef SL_SPIDRV_EXP_CLK_LOC
#define SL_SPIDRV_EXP_CLK_LOC 0 
#endif

#define ARDUCAM_USART_PERIPHERAL                SL_SPIDRV_EXP_PERIPHERAL
#define ARDUCAM_USART_PERIPHERAL_NO             SL_SPIDRV_EXP_PERIPHERAL_NO

#define ARDUCAM_USART_CLK_PORT                  SL_SPIDRV_EXP_CLK_PORT        
#define ARDUCAM_USART_CLK_PIN                   SL_SPIDRV_EXP_CLK_PIN
#define ARDUCAM_USART_CLK_LOC                   SL_SPIDRV_EXP_CLK_LOC

#define ARDUCAM_USART_RX_PORT                   SL_SPIDRV_EXP_RX_PORT        
#define ARDUCAM_USART_RX_PIN                    SL_SPIDRV_EXP_RX_PIN
#define ARDUCAM_USART_RX_LOC                    SL_SPIDRV_EXP_RX_LOC

#define ARDUCAM_USART_TX_PORT                   SL_SPIDRV_EXP_TX_PORT        
#define ARDUCAM_USART_TX_PIN                    SL_SPIDRV_EXP_TX_PIN
#define ARDUCAM_USART_TX_LOC                    SL_SPIDRV_EXP_TX_LOC

#define ARDUCAM_USART_CS_PORT                   SL_SPIDRV_EXP_CS_PORT      
#define ARDUCAM_USART_CS_PIN                    SL_SPIDRV_EXP_CS_PIN
