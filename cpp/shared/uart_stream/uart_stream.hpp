/**
 * UART Stream
 *
 * This allows for streaming binary data across a UART.
 *
 * Features:
 * - Asynchronous reception of binary data
 * - Data flow control
 * - Python library (see mltk.utils.uart_stream)
 * - Send/receive "commands"
 * - Data transferred using DMA
 *
*/
#pragma once


#include <stdint.h>

#include "uart_stream_interface.h"



namespace uart_stream
{



/***
 * Initialize the UART Stream library
 *
 * @param rx_buffer Buffer to hold asynchronous data received from other side
 * @param rx_buffer_length The size of the given rx_buffer
 * @param baud_rate Baud rate to use. Leave as 0 to use system BAUD rate
 * @return true if initialization was successful, false else
*/
bool initialize(
    uint8_t* rx_buffer,
    uint32_t rx_buffer_length,
    uint32_t baud_rate = 0
);

/**
 * Return the amount of data that can be immediately read using the @ref read() API.
 *
 * @return Amount of bytes that can be immediately read by @ref read().
 *  If -1 then the UART link has not been synchronized with the other side.
 */
int32_t get_rx_bytes_available();

/**
 * Return the amount of data that can be immediately written using the @ref write() API.
 *
 * @return Amount of bytes that can be immediately written by @ref write().
 *  If -1 then the UART link has not been synchronized with the other side.
 */
int32_t get_tx_bytes_available();

/**
 * Synchronize UART link with other size
 *
 * Before data transfer may behind, both sides need to "synchronize".
 * This should be periodically called until it returns true indicating
 * that the link is sync'd.
 *
 * @note This is non-blocking
 *
 * @return true if link is synchronized, false else
*/
bool    synchronize();

/**
 * Return if the link is sync'd
 *
 * @return true if link is synchronized, false else
*/
bool    is_synchronized();

/**
 * Write binary data
 *
 * This writes binary data using DMA.
 * Only update to @ref get_tx_bytes_available() will be written.
 *
 * @param data Binary data to write
 * @param max_length Maximum amount of data to write
 * @param block If true then this API blocks until the transfer is complete.
 *  If false, then the DMA is started and this API immediately returns.
 *  In this case, @ref get_tx_bytes_available() returns 0 until the DMA is complete.
 * @return The number of bytes written. -1 if the link is not synchronized.
*/
int32_t write(const void *data, int32_t max_length, bool block=true);

/**
 * Read binary data
 *
 * This read binary data from the rx_buffer provided during @ref initialize().
 * Only update to @ref get_rx_bytes_available() will be read.
 *
 * @param data Buffer to hold read data
 * @param max_length Maximum amount of data to read. If -1 then read @ref get_rx_bytes_available() bytes of data.
 * @return The number of bytes read. -1 if the link is not synchronized.
*/
int32_t read(void *data, int32_t max_length=-1);

/**
 * Directly read binary data
 *
 * This returns a pointer into the rx_buffer provided during @ref initialize().
 * Only update to @ref get_rx_bytes_available() are guaranteed to be available after the returned pointer.
 * The returned number of bytes is also guaranteed to be contiguous.
 *
 * @param data_ptr Buffer to hold pointer into rx_buffer
 * @param max_length Maximum amount of data to be directly read. If -1 then attempt to read @ref get_rx_bytes_available() bytes of data.
 * @return The number of contiguous bytes read (i.e. immediately available starting at the returned data_ptr). -1 if the link is not synchronized.
*/
int32_t read_direct(uint8_t** data_ptr, int32_t max_length=-1);

/**
 * Flush the RX Buffer
 *
 * Drop any data in the RX buffer
 *
 * @param max_length Maximum amount of data to be dropped. If -1 then drop @ref get_rx_bytes_available() bytes of data.
 * @return Numer of RX bytes dropped.  -1 if the link is not synchronized.
*/
int32_t flush(int32_t max_length=-1);

/**
 * Send a command to the other side
 *
 * A "command" consists of an unsigned, 8-bit code and an optional, 6-byte payload.
 * While the command is guaranteed to be sent the other side.
 * Reception at the other side's application-level is not guaranteed.
 *
 * @note Unread commands on the other side will be silently dropped
 *
 * @param code unsigned, 8-bit command code
 * @param payload optional, 6-byte payload
 * @return true if the command was sent to the other side, false else
*/
bool    write_cmd(uint8_t code, const uint8_t payload[COMMAND_PAYLOAD_LENGTH] = nullptr);

/**
 * Read a command to the other side
 *
 * A "command" consists of an unsigned, 8-bit code and an optional, 6-byte payload.
 * If the other side sends commands, then the application should periodically
 * call this API to receive the command.
 *
 * @note Unread commands will be silently dropped
 *
 * @param code unsigned, 8-bit command code
 * @param payload optional, 6-byte payload
 * @return true if a command was read, false else
*/
bool    read_cmd(uint8_t* code, uint8_t payload[COMMAND_PAYLOAD_LENGTH] = nullptr);

} // namespace uart_stream