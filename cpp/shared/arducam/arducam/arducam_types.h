#pragma once


#include "sl_status.h"


#ifdef __cplusplus
extern "C" {
#endif


#define _ARDUCAM_LEVEL_COUNT(name) (((ARDUCAM_ ## name ## _LEVEL_MAX) - (ARDUCAM_## name ##_LEVEL_MIN)) + 1)

/**
 * @brief Brightness levels
 * 
 * @ref ARDUCAM_SETTING_BRIGHTNESS min/max levels
 */
#define ARDUCAM_BRIGHTNESS_LEVEL_MIN     -2
#define ARDUCAM_BRIGHTNESS_LEVEL_MAX     2
#define ARDUCAM_BRIGHTNESS_LEVELS        _ARDUCAM_LEVEL_COUNT(BRIGHTNESS)

/**
 * @brief Contrast levels
 * 
 * @ref ARDUCAM_SETTING_CONTRAST min/max levels
 */
#define ARDUCAM_CONTRAST_LEVEL_MIN       -2
#define ARDUCAM_CONTRAST_LEVEL_MAX       2
#define ARDUCAM_CONTRAST_LEVELS          _ARDUCAM_LEVEL_COUNT(CONTRAST)

/**
 * @brief Saturation levels
 * 
 * @ref ARDUCAM_SETTING_SATURATION min/max levels
 */
#define ARDUCAM_SATURATION_LEVEL_MIN     -2
#define ARDUCAM_SATURATION_LEVEL_MAX     2
#define ARDUCAM_SATURATION_LEVELS        _ARDUCAM_LEVEL_COUNT(SATURATION)

/**
 * @brief Gain ceiling levels
 * 
 * @ref ARDUCAM_SETTING_GAINCEILING min/max levels
 */
#define ARDUCAM_GAINCEILING_MIN         0
#define ARDUCAM_GAINCEILING_MAX         6


/**
 * @brief Camera setting
 * 
 * Setting supported by @ref arducam_set_setting() API
 */
typedef enum
{
    ARDUCAM_SETTING_BRIGHTNESS,
    ARDUCAM_SETTING_CONTRAST,
    ARDUCAM_SETTING_SATURATION,
    ARDUCAM_SETTING_MIRROR,
    ARDUCAM_SETTING_FLIP,
    ARDUCAM_SETTING_SPECIALEFFECT,
    ARDUCAM_SETTING_GAINCEILING,
    ARDUCAM_SETTING_MAX
} arducam_setting_t;



/**
 * @brief Camera sensor resolution
 * 
 * Resolutions supported by camera's sensor
 */
typedef enum
{
    ARDUCAM_RESOLUTION_AUTO,
    ARDUCAM_RESOLUTION_160x120,
    ARDUCAM_RESOLUTION_176x144,
    ARDUCAM_RESOLUTION_320x240,
    ARDUCAM_RESOLUTION_352x288,
    ARDUCAM_RESOLUTION_640x480,
    ARDUCAM_RESOLUTION_800x600,
    ARDUCAM_RESOLUTION_1024x768,
    ARDUCAM_RESOLUTION_1280x1024,
    ARDUCAM_RESOLUTION_1600x1200,
} arducam_resolution_t;


/**
 * @brief Data format
 * 
 * Supported data formats
 */
typedef enum
{
    ARDUCAM_DATA_FORMAT_RGB565,
    ARDUCAM_DATA_FORMAT_RGB888,
    ARDUCAM_DATA_FORMAT_GRAYSCALE,
    ARDUCAM_DATA_FORMAT_YUV422,
} arducam_data_format_t;

/**
 * @brief Special effects
 * 
 * Special effects supported by @ref ARDUCAM_SETTING_SPECIALEFFECT setting
 */
typedef enum 
{
    ARDUCAM_SPECIALEFFECT_NONE      = 0,
    ARDUCAM_SPECIALEFFECT_NEGATIVE  = 1,
    ARDUCAM_SPECIALEFFECT_GRAY      = 2,
    ARDUCAM_SPECIALEFFECT_SEPIA     = 3,
    ARDUCAM_SPECIALEFFECT_BLUISH    = 4,
    ARDUCAM_SPECIALEFFECT_REDDISH   = 5,
    ARDUCAM_SPECIALEFFECT_GREENISH  = 6,
    ARDUCAM_SPECIALEFFECT_ANTIQUE   = 7,
    ARDUCAM_SPECIALEFFECT_COUNT     = 8,
} arducam_special_effect_t;

/**
 * @brief Driver configuration
 * 
 * Configuration values provided to @ref arducam_init().
 */
typedef struct
{
    arducam_data_format_t data_format;          //!< Data format
    arducam_resolution_t sensor_resolution;     //!< Sensor resolution
    struct
    {
        uint32_t width;                         //!< Width of output image
        uint32_t height;                        //!< Height of output image
    } image_resolution;
} arducam_config_t;

/**
 * @brief Default driver configuration
 * 
 * Default config uses:
 * - data_format = @ref ARDUCAM_DATA_FORMAT_RGB888
 * - sensor_resolution = @ref ARDUCAM_RESOLUTION_AUTO
 * - image width, height = 96x96
 */
#define ARDUCAM_DEFAULT_CONFIG \
{ \
    ARDUCAM_DATA_FORMAT_RGB888, /* data_format = RGB  */ \
    ARDUCAM_RESOLUTION_AUTO, /* sensor_resolution = AUTO */ \
    { 96, 96 } /* Image resolution = 96x96 */ \
}


#ifdef __cplusplus
}
#endif