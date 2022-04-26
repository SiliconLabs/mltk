#include <string.h>

#include "dmadrv.h"
#include "em_usart.h"
#include "r503_driver.h"
#include "fingerprint_reader_internal.h"



#define MAX_PACKET_LENGTH (sizeof(r503_packet_header_t) + R503_PACKET_MAX_LENGTH + sizeof(uint16_t))




static sl_status_t configure_baud();
static sl_status_t configure_data_length();
static uint16_t swap_endianess16(uint16_t v);
static uint32_t swap_endianess32(uint32_t v);
static sl_status_t read_packet_header(r503_packet_header_t* header, uint16_t *checksum);
static uint16_t calculate_packet_checksum(const uint8_t* packet, uint16_t data_length);
static bool validate_packet_checksum(const r503_packet_t* packet, uint16_t checksum);
static uint16_t calculate_checksum(const uint8_t* data, uint16_t data_length, uint16_t *checksum_ptr);
static sl_status_t read_data(uint8_t* data, uint16_t length);
static sl_status_t write_data(const uint8_t* buffer, uint16_t length);
 

static struct 
{
    unsigned int dma_channel_id;
    LDMA_Descriptor_t dma_desc;
    uint8_t data[MAX_PACKET_LENGTH];
    uint8_t* end;
    uint8_t *ptr;
} rx_buffer;



/*************************************************************************************************/
sl_status_t r503_init(void)
{
    sl_status_t status;

    DMADRV_Init();
    if(DMADRV_AllocateChannel(&rx_buffer.dma_channel_id, NULL) != 0)
    {
        return SL_STATUS_ALLOCATION_FAILED;
    }

    LDMA_TransferCfg_t rx_cfg = LDMA_TRANSFER_CFG_PERIPHERAL(USART_RX_DMA_SIGNAL(FINGERPRINT_READER_USART_PERIPHERAL_NO));
    LDMA_Descriptor_t rx_desc = LDMA_DESCRIPTOR_LINKREL_P2M_BYTE(
        &FINGERPRINT_READER_USART->RXDATA,
        rx_buffer.data,
        MAX_PACKET_LENGTH,
        0 // Link back to this descriptor
    );
    rx_desc.xfer.doneIfs = 0;
    memcpy(&rx_buffer.dma_desc, &rx_desc, sizeof(LDMA_Descriptor_t));
    LDMA_StartTransfer(rx_buffer.dma_channel_id, &rx_cfg, &rx_buffer.dma_desc);
    rx_buffer.end = rx_buffer.data + MAX_PACKET_LENGTH;
    rx_buffer.ptr = rx_buffer.data;


    // Configure the BAUD rate to its highest value
    status = configure_baud();
    if(status != SL_STATUS_OK)
    {
        return status;
    }

#ifdef FINGERPRINT_READER_DUMP_SENSOR_INFO
    r503_product_information_t info;
    status = r503_read_product_info(&info);
    if(status != SL_STATUS_OK)
    {
        return status;
    }

    printf("--------------------------------------\n");
    printf("Fingerprint reader information:\n");
    printf("Model: %s\n", info.model);
    printf("Batch number: %s\n", info.batch_number);
    printf("Serial number: %s\n", info.serial_number);
    printf("Hardware version: %d.%d\n", info.hw_version_major, info.hw_version_minor);
    printf("Sensor type: %s\n", info.sensor_type);
    printf("Image size: %dx%d\n", info.image_width, info.image_height);
#endif

    // Ensure the data length is configured to it maximum value
    status = configure_data_length();
    if(status != SL_STATUS_OK)
    {
        return status;
    }

    r503_status_t command_status;
    status = r503_send_command(R503_COMMAND_CHECK_SENSOR, NULL, 0, &command_status, NULL, 0);
    if(status != SL_STATUS_OK)
    {
        return status;
    }

    if(command_status != R503_STATUS_OK)
    {
        return SL_STATUS_NOT_AVAILABLE;
    }

    return status;
}

/*************************************************************************************************/
sl_status_t r503_deinit(void)
{
    DMADRV_FreeChannel(rx_buffer.dma_channel_id);

    return SL_STATUS_OK;
}


/*************************************************************************************************/
sl_status_t r503_read_system_parameters(r503_system_parameters_t *params)
{
    sl_status_t status;
    
    status = r503_send_command(R503_COMMAND_GET_SYSTEM_PARAMETERS, NULL, 0, NULL, (uint8_t*)params, sizeof(r503_system_parameters_t));
    if(status == SL_STATUS_OK)
    {
        params->status = swap_endianess16(params->status);
        params->system_id = swap_endianess16(params->system_id);
        params->finger_library_size = swap_endianess16(params->finger_library_size);
        params->security_level = swap_endianess16(params->security_level);
        params->device_address = swap_endianess32(params->device_address);
        params->data_packet_size = swap_endianess16(params->data_packet_size);
        params->baud = swap_endianess16(params->baud);
    }

    return status;
}


/*************************************************************************************************/
sl_status_t r503_read_product_info(r503_product_information_t *info)
{
    sl_status_t status;
    
    status = r503_send_command(R503_COMMAND_READ_PRODUCT_INFO, NULL, 0, NULL, (uint8_t*)info, sizeof(r503_product_information_t));
    if(status == SL_STATUS_OK)
    {
        info->model[sizeof(info->model)-1] = 0;
        info->batch_number[sizeof(info->batch_number)-1] = 0;
        info->serial_number[sizeof(info->serial_number)-1] = 0;
        info->sensor_type[sizeof(info->sensor_type)-1] = 0;
        info->image_width = swap_endianess16(info->image_width);
        info->image_height = swap_endianess16(info->image_height);
        info->template_size = swap_endianess16(info->template_size);
        info->template_total = swap_endianess16(info->template_total);
    }

    return status;
}


/*************************************************************************************************/
sl_status_t r503_update_led(const r503_led_config_t *config)
{
    return r503_send_command(R503_COMMAND_UPDATE_LED, (uint8_t*)config, sizeof(r503_led_config_t), NULL, NULL, 0);
}


/*************************************************************************************************/
sl_status_t r503_capture_image(r503_status_t *status)
{
    sl_status_t sl_status = r503_send_command(R503_COMMAND_CAPTURE_FINGERPRINT, NULL, 0, status, NULL, 0);
    if(sl_status != SL_STATUS_OK || *status != R503_STATUS_OK)
    {
        r503_cancel();
    }

    return sl_status;
}


/*************************************************************************************************/
static sl_status_t read_image_chunk(uint8_t** dst_ptr, uint16_t dst_length)
{
    sl_status_t status;

    // Read the data packet header
    uint8_t packet_buffer[MAX_PACKET_LENGTH];
    uint8_t *data_buffer = &packet_buffer[sizeof(r503_packet_header_t)];
    r503_packet_header_t* packet_header = (r503_packet_header_t*)packet_buffer;
    uint16_t header_checksum;

    status = read_packet_header(packet_header, &header_checksum);
    if(status != SL_STATUS_OK)
    {
        FPR_DEBUG("Failed to read data packet header");
        return status;
    }
    else if(!(packet_header->type == R503_PACKET_TYPE_DATA || packet_header->type == R503_PACKET_TYPE_END_OF_DATA))
    {
        FPR_DEBUG("Received packet is not a data packet");
        return SL_STATUS_INVALID_STATE;
    }

    status = read_data(data_buffer, packet_header->length);
    if(status != SL_STATUS_OK)
    {
        FPR_DEBUG("Failed to receive image data packet");
        return SL_STATUS_INVALID_STATE;
    }

    if(!validate_packet_checksum((r503_packet_t*)packet_buffer, header_checksum))
    {
        FPR_DEBUG("Failed to verify data packet checksum");
        return SL_STATUS_BUS_ERROR;
    }

    if(packet_header->type == R503_PACKET_TYPE_END_OF_DATA)
    {
        return SL_STATUS_OK;
    }

    // Each pixel is 4-bits, 2 pixels per byte, high nibble first
    const uint8_t* src = data_buffer;
    uint8_t* dst = *dst_ptr;
    for(uint16_t packet_remaining = packet_header->length-sizeof(uint16_t); packet_remaining > 0; --packet_remaining, ++src)
    {
        if(dst_length < 2)
        {
            FPR_DEBUG("Data overflow");
            return SL_STATUS_WOULD_OVERFLOW;
        }

        *dst++ = (*src & 0xF0); // NOTE: We keep the 4-bits in the upper part of the dst bytes
        *dst++ = (*src << 4);   //       to make the image appear brighter (this is the same as multiplying each pixel by 16)
        dst_length -= 2;
    }
    *dst_ptr = dst;

    return SL_STATUS_IN_PROGRESS;
}

/*************************************************************************************************/
sl_status_t r503_read_image(uint8_t* image_buffer)
{
    sl_status_t status;
    uint8_t packet_buffer[MAX_PACKET_LENGTH];

    status = R503_SEND_SIMPLE_COMMAND(R503_COMMAND_RETRIEVE_IMAGE_CHUNK);
    if(status != SL_STATUS_OK)
    {
        r503_cancel();
        return status;
    }

    uint8_t* ptr = image_buffer;
    uint8_t* image_buffer_end = ptr + FINGERPRINT_READER_IMAGE_WIDTH*FINGERPRINT_READER_IMAGE_HEIGHT;

    for(;;)
    {
        if(ptr > image_buffer_end)
        {
            status = SL_STATUS_HAS_OVERFLOWED;
            break;
        }

        uint16_t remaining = (uint16_t)(image_buffer_end - ptr);
        status = read_image_chunk(&ptr, remaining);
        if(status == SL_STATUS_IN_PROGRESS)
        {
            continue;
        }

        break;
    }

    if(status != SL_STATUS_OK)
    {
        r503_cancel();
    }

    return status;
}

/*************************************************************************************************/
sl_status_t r503_cancel()
{
    for(int i = 0; i < 10; ++i)
    {
        r503_status_t command_status;
        sl_status_t status = r503_send_command(R503_COMMAND_CANCEL, NULL, 0, &command_status, NULL, 0);
        if(status == SL_STATUS_OK && command_status == R503_STATUS_OK)
        {
            return SL_STATUS_OK;
        }
    }

    return SL_STATUS_FAIL;
}


/*************************************************************************************************/
sl_status_t r503_send_command(
    r503_command_t command, 
    const uint8_t* data, 
    uint16_t data_length, 
    r503_status_t *response_status,
    uint8_t* response_buffer,
    uint16_t response_buffer_length
)
{
    sl_status_t status;
    uint8_t packet_buffer[MAX_PACKET_LENGTH]; 
    uint8_t *data_buffer = &packet_buffer[sizeof(r503_packet_header_t)];
    r503_packet_header_t* packet_header = (r503_packet_header_t*)packet_buffer;

    packet_header->delimiter = swap_endianess16(R503_PACKET_HEADER_DELIMITER);
    packet_header->address = R503_PACKET_DEFAULT_ADDRESS;
    packet_header->type = R503_PACKET_TYPE_COMMAND;
    packet_header->length = swap_endianess16(1 + data_length + sizeof(uint16_t));

    data_buffer[0] = (uint8_t)command;

    if(data != NULL)
    {
        memcpy(&data_buffer[1], data, data_length);
    }

    uint16_t checksum = calculate_packet_checksum(packet_buffer, 1 + data_length);
    checksum = swap_endianess16(checksum);
    memcpy(&data_buffer[1 + data_length], &checksum, sizeof(uint16_t));

    write_data(packet_buffer, sizeof(r503_packet_header_t) + 1 + data_length + sizeof(uint16_t));

    uint16_t header_checksum;
    status = read_packet_header(packet_header, &header_checksum);
    if(status != SL_STATUS_OK)
    {
        return status;
    }
    else if(packet_header->type != R503_PACKET_TYPE_ACK)
    {
        return SL_STATUS_INVALID_STATE;
    }

    status = read_data(data_buffer, packet_header->length);
    if(status != SL_STATUS_OK)
    {
        return status;
    }
    if(!validate_packet_checksum((r503_packet_t*)packet_buffer, header_checksum))
    {
        return SL_STATUS_BUS_ERROR;
    }


    r503_status_t command_status = data_buffer[0];
    if(command_status == R503_STATUS_RX_ERROR || 
       command_status == R503_STATUS_COMM_ERROR || 
       command_status == R503_STATUS_TRANSFER_ERROR)
    {
        FPR_DEBUG("command status err: %d", command_status);
        return SL_STATUS_BUS_ERROR;
    }

    if(response_status != NULL)
    {
        *response_status = command_status;
    }

    if(response_buffer != NULL)
    {
        memcpy(response_buffer, &data_buffer[1], response_buffer_length);
    }

    return SL_STATUS_OK;
}

/*************************************************************************************************/
static sl_status_t read_packet_header(r503_packet_header_t* header, uint16_t *checksum)
{
    sl_status_t status = read_data((uint8_t*)header, sizeof(r503_packet_header_t));
    if(status != SL_STATUS_OK)
    {
        return status;
    }

    *checksum = calculate_packet_checksum((uint8_t*)header, 0);

    header->delimiter = swap_endianess16(header->delimiter);
    header->address = swap_endianess32(header->address);
    header->length = swap_endianess16(header->length);

    return SL_STATUS_OK;
}


/*************************************************************************************************/
static uint16_t calculate_packet_checksum(const uint8_t* packet, uint16_t data_length)
{
    return calculate_checksum(packet + sizeof(uint16_t) + sizeof(uint32_t), 3 + data_length, NULL);
}


/*************************************************************************************************/
static bool validate_packet_checksum(const r503_packet_t* packet, uint16_t header_checksum)
{
    const uint16_t data_length = packet->length - sizeof(uint16_t);
    const uint16_t checksum = calculate_checksum(packet->data, data_length, &header_checksum);

    uint16_t expected_checksum;
    memcpy(&expected_checksum, &packet->data[data_length], sizeof(uint16_t));
    expected_checksum = swap_endianess16(expected_checksum);

    return expected_checksum == checksum;
}


/*************************************************************************************************/
static uint16_t calculate_checksum(const uint8_t* data, uint16_t data_length, uint16_t *checksum_ptr)
{
    uint16_t checksum = (checksum_ptr == NULL) ? 0 : *checksum_ptr;
    for(uint16_t remaining = data_length; remaining > 0; --remaining)
    {
        checksum += *data++;
    }
    return checksum;
}


/*************************************************************************************************/
static sl_status_t read_data(uint8_t* buffer, uint16_t length)
{
#define current_dma_ptr() (rx_buffer.data + (MAX_PACKET_LENGTH - LDMA_TransferRemainingCount(rx_buffer.dma_channel_id)))
    uint8_t* dst = buffer;
    
    for(uint16_t remaining = length; remaining > 0; --remaining)
    {
        for(uint32_t i = 1000000; rx_buffer.ptr == current_dma_ptr(); --i)
        {
            if(i == 0)
            {
                return SL_STATUS_TIMEOUT;
            }
        }

        *dst++ = *rx_buffer.ptr++;
        if(rx_buffer.ptr == rx_buffer.end)
        {
            rx_buffer.ptr = rx_buffer.data;
        }
    }

    return SL_STATUS_OK;
}

/*************************************************************************************************/
static sl_status_t write_data(const uint8_t* buffer, uint16_t length)
{
    const uint8_t* src = buffer;
    for(uint16_t remaining = length; remaining > 0; --remaining)
    {
        while (!(FINGERPRINT_READER_USART->STATUS & USART_STATUS_TXBL))
        {
        }
        FINGERPRINT_READER_USART->TXDATA = (uint32_t)*src++;
    }

    while (!(FINGERPRINT_READER_USART->STATUS & USART_STATUS_TXBL))
    {
    }
    return SL_STATUS_OK;
}


/*************************************************************************************************/
static sl_status_t configure_baud()
{
    sl_status_t status;

    // First try to read the sensor at 115200
    USART_BaudrateAsyncSet(FINGERPRINT_READER_USART, 0, 115200, usartOVS16);
    status = R503_SEND_SIMPLE_COMMAND(R503_COMMAND_HANDSHAKE);

    // If the command failed, then try using the default baud rate
    if(status != SL_STATUS_OK)
    {
        USART_BaudrateAsyncSet(FINGERPRINT_READER_USART, 0, 57600, usartOVS16);
        status = R503_SEND_SIMPLE_COMMAND(R503_COMMAND_HANDSHAKE);
        if(status != SL_STATUS_OK)
        {
            return SL_STATUS_INITIALIZATION;
        }

        // The sensor is currently configured for 57600 baud
        // Update the baud to 115200
        const uint8_t data[2] = {R503_PARAMETER_BAUD, R503_BAUD_115200};
        status = r503_send_command(R503_COMMAND_SET_SYSTEM_PARAMETER, data, sizeof(data), NULL, NULL, 0);
        if(status != SL_STATUS_OK)
        {
            return status;
        }

        USART_BaudrateAsyncSet(FINGERPRINT_READER_USART, 0, 115200, usartOVS16);

        // short delay while baud rate changes on reader
        DELAY_MS(50);
 
        status = R503_SEND_SIMPLE_COMMAND(R503_COMMAND_HANDSHAKE);
    }

    return status;
}


/*************************************************************************************************/
static sl_status_t configure_data_length()
{
    sl_status_t status;

    // Read the system parameters
    r503_system_parameters_t params;
    status = r503_read_system_parameters(&params);
    if(status != SL_STATUS_OK)
    {
        return status;
    }

    // Ensure the data length is at its max value
    if(params.data_packet_size != R503_DATA_LENGTH_256)
    {
        const uint8_t data[2] = {R503_PARAMETER_DATA_LENGTH, R503_DATA_LENGTH_256};
        status = r503_send_command(R503_COMMAND_SET_SYSTEM_PARAMETER, data, 2, NULL, NULL, 0);
        if(status != SL_STATUS_OK)
        {
            return status;
        }
    }

    return status;
}

/*************************************************************************************************/
static uint16_t swap_endianess16(uint16_t v)
{
    return (v >> 8) | (v << 8); 
}

/*************************************************************************************************/
static uint32_t swap_endianess32(uint32_t v)
{
  uint8_t input[4];
  memcpy(input, &v, sizeof(uint32_t));
  return (
    (((uint32_t) input[0]) << 24) |
    (((uint32_t) input[1]) << 16) |
    (((uint32_t) input[2]) <<  8) |
    (((uint32_t) input[3]) <<  0)
    );
}

