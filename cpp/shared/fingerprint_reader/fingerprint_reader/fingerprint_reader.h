#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "sl_status.h"

#ifdef __cplusplus
extern "C" {
#endif



#define FINGERPRINT_READER_IMAGE_WIDTH 192
#define FINGERPRINT_READER_IMAGE_HEIGHT 192



typedef struct
{
    void (*finger_detected_irq_callback)(void);
} fingerprint_reader_config_t;

#define FINGERPRINT_READER_DEFAULT_CONFIG \
{ \
    /*finger_detected_irq_callback*/nullptr \
}


typedef enum 
{
    fprRedLed = 1,
    fprBlueLed = 2,
    fprPurpleLed = 3
} fingerprint_reader_led_color_t;

typedef enum 
{
    fprBreathingMode = 1,
    fprFlashingMode = 2,
    fprAlwaysOnMode = 3,
    fprAlwaysOffMode = 4,
    fprGradualOnMode = 5,
    fprGradualOffMode = 6,
} fingerprint_reader_led_mode_t;

typedef struct 
{
    fingerprint_reader_led_color_t color;      //!< LED color
    fingerprint_reader_led_mode_t mode;        //!< Display mode
    uint8_t speed;              //!< Mode speed, 0-256, minimum o 5s cycles
                                //!< This is the effective breakthing/flashing/gradual on/gradual off duration
    uint8_t count;              //!< Number of time mode should repeat. 0 = infinite, 1-255       
} fingerprint_reader_led_config_t;


typedef uint8_t fingerprint_reader_image_t[FINGERPRINT_READER_IMAGE_WIDTH*FINGERPRINT_READER_IMAGE_HEIGHT];



sl_status_t fingerprint_reader_init(const fingerprint_reader_config_t* config);

sl_status_t fingerprint_reader_deinit();

sl_status_t fingerprint_reader_update_led(const fingerprint_reader_led_config_t* config);

bool fingerprint_reader_is_image_available();

sl_status_t fingerprint_reader_get_image(fingerprint_reader_image_t image_buffer);



#ifdef __cplusplus
}
#endif