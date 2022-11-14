#pragma once

#include <stdint.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    uint8_t *buffer;                ///< Pointer to allocated buffer
    const uint8_t *buffer_end;      ///< Pointer to end of allocated buffer
    uint8_t *append;                ///< Pointer to where data should be appended to the buffer
    uint8_t *prepend;               ///< Pointer to where data should be prepended to the buffer
} dynamic_buffer_t;



/**
 * Allocate dynamic buffer
 *
 * This allocates a dynamic buffer of the specified length
 *
 * @param[in] buffer @ref dynamic_buffer_t object to allocate a buffer for
 * @param[in] length Length in bytes of allocated buffer
 * @return @ref 0 on success
 */
int dynamic_buffer_alloc(dynamic_buffer_t *buffer, uint32_t length);

/**
 * Re-allocate dynamic buffer
 *
 * This allocates a larger buffer if necessary. After this API returns,
 * the given buffer will have at least `additional_length` of additional space.
 *
 * @note The contents of the previous buffer will remain unchanged.
 *
 * @param[in] buffer @ref dynamic_buffer_t object to allocate a larger buffer (if necessary)
 * @param[in] additional_length Additional required size
 * @return @ref 0 on success
 */
int dynamic_buffer_realloc(dynamic_buffer_t *buffer, uint32_t additional_length);

/**
 * Copy dynamic buffer
 *
 * This copies the contents of the `src` @ref dynamic_buffer_t into the `dst` buffer.
 *
 * @param[in] dst Destination buffer to receive copied data from `src`
 * @param[in] src Source buffer to copy to `dst`
 * @return @ref 0 on success
 */
int dynamic_buffer_copy(dynamic_buffer_t *dst, const dynamic_buffer_t *src);

/**
 * De-allocate dynamic buffer
 *
 * This de-allocates the memory used by a @ref dynamic_buffer_t
 *
 * @param[in] buffer Buffer to de-allocate memory
 */
void dynamic_buffer_free(dynamic_buffer_t *buffer);

/**
 * Reset buffer pointers
 *
 * This resets the @ref dynamic_buffer_t `append` and `prepend` pointer
 * to the beginning of the allocated buffer.
 *
 * @param[in] buffer Buffer to reset
 */
void dynamic_buffer_reset(dynamic_buffer_t *buffer);

/**
 * Write data to dynamic buffer
 *
 * This write data to a @ref dynamic_buffer_t. If the amount of data to
 * write is larger than the buffer's current size then more buffer will be allocated as necessary.
 *
 * @param[in] buffer Buffer to write
 * @param[in] data Data to write to `buffer`
 * @param[in] data_length Size in bytes of `data
 * @return @ref 0 on success
 */
int dynamic_buffer_write(dynamic_buffer_t *buffer, const void *data, uint32_t data_length);

/**
 * Write formatted string to buffer
 *
 * This writes a formatted string to the @ref dynamic_buffer_t.
 * The @ref dynamic_buffer_t buffer size will increase as necessary.
 *
 * @param[in] buffer Buffer to write
 * @param[in] fmt Formatted string
 * @return @ref 0 on success
 */
int dynamic_buffer_writef(dynamic_buffer_t *buffer, const char *fmt, ...);

/**
 * Write formatted string to buffer
 *
 * This writes a formatted string to the @ref dynamic_buffer_t.
 * The @ref dynamic_buffer_t buffer size will increase as necessary.
 *
 * @param[in] buffer Buffer to write
 * @param[in] fmt Formatted string
 * @param[in] args Formatted string arguments
 * @return @ref 0 on success
 */
int dynamic_buffer_vwrite(dynamic_buffer_t *buffer, const char *fmt, va_list args);

/**
 * Get populated data length
 *
 * This returns the populated data length of the buffer.
 * e.g.:
 * @code{.c}
 * length = buffer->append - buffer->prepend
 * @endcode
 *
 * @param[in] buffer @ref dynamic_buffer_t
 * @return Length of populated data
 */
uint32_t dynamic_buffer_get_length(const dynamic_buffer_t *buffer);

/**
 * Get allocated buffer size
 *
 * This return the total size of the @ref dynamic_buffer_t allocated buffer.
 * e.g.:
 * @code{.c}
 * size = buffer->buffer_end - buffer->buffer
 * @endcode
 *
 * @param[in] buffer @ref dynamic_buffer_t
 * @return Size of allocated buffer
 */
uint32_t dynamic_buffer_get_total_size(const dynamic_buffer_t *buffer);

/**
 * Get amount remaining in allocated buffer
 *
 * This returns the number of unpopulated bytes in the allocated buffer.
 * e.g.:
 * @code{.c}
 * length = buffer->buffer_end - buffer->append
 * @endcode
 *
 * @param[in] buffer @ref dynamic_buffer_t
 * @return Bytes remaining in allocated buffer
 */
uint32_t dynamic_buffer_get_remaining_length(const dynamic_buffer_t *buffer);

/**
 * Get buffer contents
 *
 * This returns a pointer to the buffer and populated size.
 * e.g.:
 * @code{.c}
 * buffer->data = dynamic_buffer->prepend
 * buffer->size = buffer->append - buffer->prepend
 * @endcode
 *
 * @param[in] dynamic_buffer @ref dynamic_buffer_t
 * @param[out] buffer_ptr Pointer to hold underlying buffer
 * @param[out] length_ptr Pointer to size of populated buffer
 * @return @ref 0 on success
 */
int dynamic_buffer_get_buffer(const dynamic_buffer_t *dynamic_buffer, uint8_t** buffer_ptr, uint32_t* length_ptr);

/**
 * Adjust append pointer of buffer
 *
 * This moves the `append` pointer forwards or backwards by `(signed)adjust_amount`
 *
 * @param[in] buffer @ref dynamic_buffer_t
 * @param[in] adjust_amount Amount to move `append` pointer
 * @return @ref 0 on success
 */
int dynamic_buffer_adjust_data_end(dynamic_buffer_t *buffer, int32_t adjust_amount);

/**
 * Adjust prepend pointer of buffer
 *
 *
 * This moves the `prepend` pointer forwards or backwards by `(signed)adjust_amount`
 *
 * @param[in] buffer @ref dynamic_buffer_t
 * @param[in] adjust_amount Amount to move `prepend` pointer
 * @return @ref 0 on success
 */
int dynamic_buffer_adjust_data_start(dynamic_buffer_t *buffer, int32_t adjust_amount);

/**
 * Get the data start pointer
 *
 * @param[in] buffer @ref dynamic_buffer_t
 * @return Pointer to start of data buffer
 */
static inline uint8_t* dynamic_buffer_get_data_start(const dynamic_buffer_t *buffer)
{
    return ( uint8_t*)buffer->prepend;
}



#ifdef __cplusplus
}
#endif