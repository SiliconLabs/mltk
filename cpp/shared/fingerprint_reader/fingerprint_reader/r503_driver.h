#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "sl_status.h"

#ifdef __cplusplus
extern "C" {
#endif


// #define __PACKED_STRUCT struct
// #ifndef uint8_t
// #define uint8_t unsigned char
// #endif
// #ifndef uint32_t
// #define uint32_t unsigned int
// #endif


#ifndef   __PACKED_STRUCT
  #define __PACKED_STRUCT  struct __attribute__((packed, aligned(1)))
#endif


#define R503_PACKET_HEADER_DELIMITER         0xEF01
#define R503_PACKET_DEFAULT_ADDRESS          0xFFFFFFFF
#define R503_PACKET_MAX_LENGTH               256 

typedef enum 
{
    R503_PACKET_TYPE_COMMAND   = 0x01, 
    R503_PACKET_TYPE_DATA      = 0x02,
    R503_PACKET_TYPE_ACK       = 0x07,
    R503_PACKET_TYPE_END_OF_DATA = 0x08, 
} r503_packet_type_t;


typedef __PACKED_STRUCT
{
    uint16_t delimiter;
    uint32_t address;
    r503_packet_type_t type;
    uint16_t length;
} r503_packet_header_t;

typedef __PACKED_STRUCT
{
    uint16_t header;
    uint32_t address;
    r503_packet_type_t type;
    uint16_t length;
    uint8_t data[];
} r503_packet_t;


typedef enum 
{
    R503_STATUS_OK                   = 0x00, // commad execution complete;
    R503_STATUS_RX_ERROR             = 0x01, // error when receiving data package;
    R503_STATUS_NO_FINGER            = 0x02, // no finger on the sensor;
    R503_STATUS_SENSOR_READ_ERROR    = 0x03, // fail to enroll the finger;
    R503_STATUS_OVER_IDS             = 0x06,
    R503_STATUS_BAD_IMAGE_QUALITY    = 0x07,
    R503_STATUS_BUSY                 = 0x0E, // Module can’t receive the following data packages.
    R503_STATUS_TRANSFER_ERROR       = 0x0F,
    R503_STATUS_NO_IMAGE             = 0x15,
    R503_STATUS_INVALAID_REGISTER    = 0x1A, // invalid register number;
    R503_STATUS_INVALID_CONFIG       = 0x1B, // incorrect configuration of register;
    R503_STATUS_COMM_ERROR           = 0x1D, // fail to operate the communication port;
    R503_STATUS_SENSOR_ERROR         = 0x29,
    R503_STATUS_UNKNOWN              = 0xFF
} r503_status_t;


typedef enum
{
    R503_COMMAND_SET_SYSTEM_PARAMETER        = 0x0E,
    R503_COMMAND_GET_SYSTEM_PARAMETERS       = 0x0F,
    R503_COMMAND_CAPTURE_FINGERPRINT         = 0x01,
    R503_COMMAND_RETRIEVE_IMAGE_CHUNK        = 0x0A,
    R503_COMMAND_CANCEL                      = 0x30,
    R503_COMMAND_UPDATE_LED                  = 0x35,
    R503_COMMAND_CHECK_SENSOR                = 0x36,
    R503_COMMAND_READ_PRODUCT_INFO           = 0x3C,
    R503_COMMAND_RESET                       = 0x3D,
    R503_COMMAND_HANDSHAKE                   = 0x40,
} r503_command_t;


typedef __PACKED_STRUCT
{
    char model[16];
    char batch_number[4];
    char serial_number[8];
    uint8_t hw_version_major;
    uint8_t hw_version_minor;
    char sensor_type[8];
    uint16_t image_width;
    uint16_t image_height;
    uint16_t template_size;
    uint16_t template_total;
} r503_product_information_t;


typedef __PACKED_STRUCT
{
    uint16_t status;
#define SYSTEM_STATUS_BUSY_FLAG             (1 << 0)
#define SYSTEM_STATUS_IMAGE_AVAILABLE_FLAG  (1 << 3)
    uint16_t system_id;
    uint16_t finger_library_size;
    uint16_t security_level;
    uint32_t device_address;
    uint16_t data_packet_size;
    uint16_t baud;
} r503_system_parameters_t;


typedef __PACKED_STRUCT
{
    uint8_t mode;
    uint8_t speed;
    uint8_t color;
    uint8_t count;
} r503_led_config_t;

typedef enum
{
    R503_BAUD_9600     = 1,
    R503_BAUD_19200    = 2,
    R503_BAUD_38400    = 4,
    R503_BAUD_57600    = 6,
    R503_BAUD_115200   = 12
} r503_baud_t;

typedef enum
{
    R503_DATA_LENGTH_32     = 0,
    R503_DATA_LENGTH_64     = 1,
    R503_DATA_LENGTH_128    = 2,
    R503_DATA_LENGTH_256    = 3,
} r503_data_length_t;



typedef enum
{
    R503_PARAMETER_BAUD             = 4,
    R503_PARAMETER_SECURITY_LEVEL   = 5,
    R503_PARAMETER_DATA_LENGTH      = 6,
} r503_system_parameter_t;




sl_status_t r503_init(void);
sl_status_t r503_deinit(void);
sl_status_t r503_read_system_parameters(r503_system_parameters_t *params);
sl_status_t r503_read_product_info(r503_product_information_t *info);
sl_status_t r503_update_led(const r503_led_config_t *config);
sl_status_t r503_capture_image(r503_status_t *status);
sl_status_t r503_cancel();
sl_status_t r503_read_image(uint8_t* image_buffer);
sl_status_t r503_send_command(
    r503_command_t command, 
    const uint8_t* data, 
    uint16_t data_length, 
    r503_status_t *response_status,
    uint8_t* response_buffer,
    uint16_t response_buffer_length
);

#define R503_SEND_SIMPLE_COMMAND(code) r503_send_command(code, NULL, 0, NULL, NULL, 0)


#ifdef __cplusplus
}
#endif