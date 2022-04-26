
#pragma once


#include "sl_status.h"
#include "arducam/arducam_types.h"


#ifdef __cplusplus
extern "C" {
#endif


/**
 * @brief Initialize driver
 * 
 * Initialize the driver with the given image buffer.
 * The given image buffer length should be a multiple of the output of @ref arducam_calculate_image_buffer_length().
 * Providing 2 buffers allows for best performance.
 * 
 * @note This will fail if the camera hardware fails to initialize or if the given image buffer is too small.
 * 
 * @param config @ref arducam_config_t driver configuration
 * @param image_buffer Image data buffer
 * @param buffer_length Size of the image data buffer in bytes, this should be a multiple of  @ref arducam_calculate_image_buffer_length()
 * @return sl_status_t 
 */
sl_status_t arducam_init(const arducam_config_t* config, uint8_t* image_buffer, uint32_t buffer_length);

/**
 * @brief De-initialize diver
 * 
 * @return sl_status_t 
 */
sl_status_t arducam_deinit();

/**
 * @brief Set a camera setting
 * 
 * This sets a specific camera setting. See @ref arducam_setting_t for the available settings.
 * @note Settings may be set while capturing is active.
 * 
 * @param setting @ref arducam_setting_t Setting to set
 * @param value Value of the setting
 * @return sl_status_t 
 */
sl_status_t arducam_set_setting(arducam_setting_t setting, int32_t value);
//sl_status_t arducam_get_setting(arducam_setting_t setting, int32_t* value_ptr);

/**
 * @brief Start image capturing
 * 
 * This enables the camera to begin capturing images.
 * After calling this, periodically call @ref arducam_get_next_image() to retrieve a pointer
 * to the next available image.
 * 
 * Call @ref arducam_stop_capture() to disable capturing.
 * 
 * @note @ref arducam_init() must be called before using this API.
 * 
 * @return sl_status_t 
 */
sl_status_t arducam_start_capture();

/**
 * @brief Stop image capturing
 * 
 * This disables the camera from capturing images.
 * 
 * @return sl_status_t 
 */
sl_status_t arducam_stop_capture();

/**
 * @brief Poll the camera
 * 
 * The polls the camera.
 * 
 * @note This API is optional. It can help improve throughput when
 * periodically called from a timer interrupt.
 * 
 * @note @ref arducam_start_capture() must be called before using this API.
 * 
 * @return sl_status_t 
 */
sl_status_t arducam_poll();

/**
 * @brief Attempt to retrieve the next capturing image
 * 
 * This attempts to retrieve the next capturing image from the camera.
 * If an image is available, the data_ptr argument will point to the image buffer
 * and the length_ptr will contain the length of the image in bytes.
 * 
 * This returns SL_STATUS_IN_PROGRESS if no image is currently available.
 * 
 * @ref arducam_release_image() MUST be called once the image is no longer
 * used. This API must NOT be called again until the previous image is released.
 * 
 * @note @ref arducam_start_capture() must be called before using this API.
 * 
 * @param data_ptr Pointer to hold reference to next captured image, NULL if no image is available
 * @param length_ptr Optional pointer to hold length of image in bytes. Leave NULL if unused.
 * @return sl_status_t 
 */
sl_status_t arducam_get_next_image(uint8_t** data_ptr, uint32_t* length_ptr);

/**
 * @brief Release previous image
 * 
 * This releases an image previously returned by @ref arducam_get_next_image().
 * This MUST be called once the image returned by  @ref arducam_get_next_image()
 * is no longer used.
 * 
 * @return sl_status_t 
 */
sl_status_t arducam_release_image();

/**
 * @brief Return number bytes per image
 * 
 * This returns the numbers of bytes per image used by the image buffer.
 * 
 * @note This value may be different than the value returned by @ref arducam_calculate_image_size()
 * due to internal color space conversion requirements.
 * 
 * @param format @ref arducam_data_format_t Image data format
 * @param width Width of image in pixels
 * @param height Height of image in pixels
 * @return uint32_t Bytes required given image parameters
 */
uint32_t arducam_calculate_image_buffer_length(arducam_data_format_t format, uint32_t width, uint32_t height);

/**
 * @brief Return size of image in pixels
 * 
 * This returns the size of the image in bytes.
 * 
 * @param format @ref arducam_data_format_t Image data format
 * @param width Width of image in pixels
 * @param height Height of image in pixels
 * @return uint32_t Size of image in pixels
 */
uint32_t arducam_calculate_image_size(arducam_data_format_t format, uint32_t width, uint32_t height);


#ifdef __cplusplus
}
#endif