
#pragma once

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>

#include "sl_status.h"
#include "sl_sleeptimer.h"
#include "sl_udelay.h"

// This provides the platform-specific defines
#include "arducam_config.h"

#include "arducam/arducam_types.h"
#include "ov2640.h"

#define ARDUCAM_DELAY_US(us) sl_udelay_wait(us)
#define ARDUCAM_DELAY_MS(us) sl_sleeptimer_delay_millisecond(us)



//#define ARDUCAM_DEBUG_ENABLED
#ifdef ARDUCAM_DEBUG_ENABLED
#define ARDUCAM_DEBUG(fmt, ...) printf("ARDUCAM: " fmt "\n", ## __VA_ARGS__)
#else
#define ARDUCAM_DEBUG(...)
#endif


#define ARRAY_COUNT(x) (sizeof(x) / sizeof(*x))

#ifndef MIN
#define MIN(x,y)  ((x) < (y) ? (x) : (y))
#endif /* ifndef MIN */
#ifndef MAX
#define MAX(x,y)  ((x) > (y) ? (x) : (y))
#endif /* ifndef MAX */

#define SL_VERIFY(status) if(status != SL_STATUS_OK) return status

#define MAX_IMAGE_SIZE 16721923

// Registers specific to the ArduCAM-M-2MP camera shield SPI interface


#define ARDUCHIP_RW_FLAG            0x80 // 0 is for read, 1 is for write

#define ARDUCHIP_REG_TEST1          0x00  // TEST register

#define ARDUCHIP_REG_FRAMES         0x01  // FRAME control register, Bit[2:0] = Number of frames to be captured
                                          // On 5MP_Plus platforms bit[2:0] = 7 means continuous capture until frame buffer is full


#define ARDUCHIP_REG_MODE           0x02  // Mode register
#define MCU2LCD_MODE                0x00
#define CAM2LCD_MODE                0x01
#define LCD2MCU_MODE                0x02

#define ARDUCHIP_REG_TIM            0x03  // Timming control
#define HREF_LEVEL_MASK             0x01  // 0 = High active ,       1 = Low active
#define VSYNC_LEVEL_MASK            0x02  // 0 = High active ,       1 = Low active
#define LCD_BKEN_MASK               0x04  // 0 = Enable,                     1 = Disable
#define PCLK_DELAY_MASK             0x08  // 0 = data no delay,      1 = data delayed one PCLK
#define MODE_MASK                   0x10  // 0 = LCD mode,               1 = FIFO mode
#define FIFO_PWRDN_MASK             0x20  // 0 = Normal operation, 1 = FIFO power down
#define LOW_POWER_MODE              0x40  // 0 = Normal mode,          1 = Low power mode

#define ARDUCHIP_REG_FIFO           0x04  // FIFO and I2C control
#define FIFO_CLEAR_MASK             0x01  // clear FIFO write done flag
#define FIFO_START_MASK             0x02  // start catpure
#define FIFO_RDPTR_RST_MASK         0x10  // reset FIFO write pointer
#define FIFO_WRPTR_RST_MASK         0x20  // reset FIFO read pointer

#define ARDUCHIP_REG_GPIO_DIR       0x05  // GPIO Direction Register
#define GPIO_DIR_SENSOR_RESET       0x01  // Sensor reset IO direction, 0 = input, 1 = output
#define GPIO_DIR_SENSOR_PWR_DOWN    0x02  // Sensor power down IO direction
#define GPIO_DIR_SENSOR_PWR_ENABLE  0x03  // Sensor power enable IO direction

#define ARDUCHIP_REG_GPIO_WRITE     0x06  // GPIO Write Register
#define GPIO_RESET_MASK             0x01  //0 = Sensor reset,               1 =  Sensor normal operation
#define GPIO_PWDN_MASK              0x02  //0 = Sensor normal operation,    1 = Sensor standby
#define GPIO_PWREN_MASK             0x04    //0 = Sensor LDO disable,       1 = sensor LDO enable

#define ARDUCHIP_REG_GPIO_READ      0x45

#define BURST_FIFO_READ             0x3C  // Burst FIFO read operation
#define SINGLE_FIFO_READ            0x3D  // Single FIFO read operation

#define ARDUCHIP_REG_REV            0x40  //ArduCHIP revision
#define VER_LOW_MASK                0x3F
#define VER_HIGH_MASK               0xC0

#define ARDUCHIP_REG_STATUS         0x41  // Trigger source
#define VSYNC_MASK                  0x01
#define SHUTTER_MASK                0x02
#define CAP_DONE_MASK               0x08

#define FIFO_SIZE1                  0x42  //Camera write FIFO size[7:0] for burst to read
#define FIFO_SIZE2                  0x43  //Camera write FIFO size[15:8]
#define FIFO_SIZE3                  0x44  //Camera write FIFO size[18:16]


typedef enum 
{
    CAMERA_STATE_IDLE,
    CAMERA_STATE_CAPTURING,
    CAMERA_STATE_CAPTURE_COMPLETE,
    CAMERA_STATE_READING,
    CAMERA_STATE_READ_COMPLETE
} camera_state_t;

typedef struct 
{
    uint32_t image_size_bytes;              // The length of the output image in bytes
    arducam_data_format_t data_format;      // Image data format

    struct 
    {
        uint8_t* start;                     // Start of the image buffer given to  arducam_init()
        uint8_t* end;                       // End of the image buffer given to  arducam_init()
        uint8_t* head;                      // Next image buffer address to write in the local buffer (Camera FIFO -> SPI -> head)
        uint8_t* tail;                      // Next image buffer to return to app 
        volatile uint32_t local_count;      // Number of images buffered on the device, ready to be used by the firmware
        uint32_t read_length;               // The number of image bytes to read from the camera
        uint32_t read_buffer_offset;        // The offset into the read buffer (this is required so we can expand RGB88 images in-place)
        uint32_t max_local_count;           // Maximum number of images we can store locally
        uint32_t buffer_bytes_per_image;
    } buffer;

    bool is_initialized;                    // Has the driver been initialized?
    bool is_image_locked;                   // Is am image currently being used by the application?
    volatile bool is_started;               // Has image capturing been enabled?
    volatile camera_state_t state;   
    volatile bool is_spi_active;            // Is the MCU actively using the SPI bus?

    bool add_dummy_byte_to_burst_read;      // Older Arducam versions do not require a dummy byte between the burst read command and RX data
    unsigned int dma_rx_channel;            // SPI DMA RX channel
    unsigned int dma_tx_channel;            // SPI DMA Tx channel    
    uint32_t dma_length_remaining;
    uint8_t* dma_rx_ptr;

} arducam_driver_context_t;

extern arducam_driver_context_t arducam_context;


#define REG_ADDR_ACTION 0x00
#define REG_ACTION_TERMINATE 0xFF
#ifndef   __PACKED_STRUCT
  #define __PACKED_STRUCT                        struct __attribute__((packed, aligned(1)))
#endif

typedef __PACKED_STRUCT
{
    const uint8_t address;
    const uint8_t value;
} reg_addr_value_t;


sl_status_t arducam_driver_init(const arducam_config_t* config);
sl_status_t arducam_driver_deinit();

sl_status_t arducam_driver_i2c_write_reg(uint8_t addr, uint8_t data);
sl_status_t arducam_driver_i2c_read_reg(uint8_t addr, uint8_t *val);
sl_status_t arducam_driver_i2c_write_regs(const reg_addr_value_t *regs, const reg_addr_value_t *action_list, uint8_t action_list_len);

sl_status_t arducam_driver_spi_write_reg(uint8_t addr, uint8_t data);
sl_status_t arducam_driver_spi_clear_bit(uint8_t addr, uint8_t bits);
sl_status_t arducam_driver_spi_set_bit(uint8_t addr, uint8_t bits);
sl_status_t arducam_driver_spi_read_reg(uint8_t addr, uint8_t *data_ptr);
sl_status_t arducam_driver_spi_burst_read(uint8_t *buffer, uint32_t length);
sl_status_t arducam_driver_get_fifo_size(uint32_t *size_ptr);
