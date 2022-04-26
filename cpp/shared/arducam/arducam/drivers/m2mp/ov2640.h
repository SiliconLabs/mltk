
#pragma once


#include <stdint.h>
#include "sl_status.h"



#define OV2640_ID 0x26
#define OV2640_I2C_ADDRESS 0x30


typedef struct
{
    uint8_t Manufacturer_ID1;
    uint8_t Manufacturer_ID2;
    uint8_t PIDH;
    uint8_t PIDL;
} ov2640_id_t;


sl_status_t ov2640_init(const arducam_config_t *config);
sl_status_t ov2640_deinit();
sl_status_t ov2640_set_setting(arducam_setting_t setting, int32_t value);
sl_status_t ov2640_get_setting(arducam_setting_t setting, int32_t *value_ptr);

//sl_status_t ov2640_set_zoom_and_pan(int zoom, int hpan, int vpan, int x_dither, int y_dither);
