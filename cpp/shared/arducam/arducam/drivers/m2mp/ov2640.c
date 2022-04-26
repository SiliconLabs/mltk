

#include "arducam_m_2mp_driver.h"


#include "ov2640.h"
#include "ov2640_regs.h"


#define MAX_FIFO_SIZE       0x5FFFF         //384KByte


#define OV2640_WRITE_VERIFY(addr, val) SL_VERIFY(arducam_driver_i2c_write_reg(addr, val))


// #define DEBUG_REG_ENABLE
#ifdef DEBUG_REG_ENABLE
#include <stdio.h>
#define PRINT_REG(reg, val) printf("0x%02X : 0x%02X\n", reg, val)
#else
#define PRINT_REG(reg, val)
#endif

#define WRITE_REG(reg, val) \
    PRINT_REG(reg, val); \
    SL_VERIFY(arducam_driver_i2c_write_reg(reg, val))



static sl_status_t check_comms_interface(void);
static sl_status_t set_contrast(int level);
static sl_status_t set_saturation(int level);
static sl_status_t set_brightness(int level);
static sl_status_t set_format(int format);
static sl_status_t set_gainceiling(int gainceiling);
static sl_status_t set_flip(int flip);
static sl_status_t set_mirror(int mirror);
static sl_status_t set_specialeffect(int effect);
static sl_status_t set_framesize(arducam_resolution_t resolution, uint32_t width, uint32_t height);




/*************************************************************************************************/
sl_status_t ov2640_init(const arducam_config_t *config)
{
    sl_status_t status;


    status = check_comms_interface();
    if(status != SL_STATUS_OK)
    {
        return status;
    }

    // put the sensor into reset
    OV2640_WRITE_VERIFY(BANK_SEL, BANK_SEL_SENSOR);
    OV2640_WRITE_VERIFY(COM7, COM7_SRST);
    ARDUCAM_DELAY_MS(100);

    SL_VERIFY(arducam_driver_i2c_write_regs(ov2640_default_regs, NULL, 0));
    SL_VERIFY(set_format(config->data_format));
    SL_VERIFY(set_framesize(config->sensor_resolution, config->image_resolution.width, config->image_resolution.height));

    OV2640_WRITE_VERIFY(BANK_SEL, BANK_SEL_SENSOR);
    OV2640_WRITE_VERIFY(COM10, 0x00);

    return SL_STATUS_OK;
}

/*************************************************************************************************/
sl_status_t ov2640_deinit()
{
    return SL_STATUS_OK;
}


/*************************************************************************************************/
sl_status_t ov2640_set_setting(arducam_setting_t setting, int32_t value)
{
    sl_status_t result;

    switch(setting)
    {
    case ARDUCAM_SETTING_BRIGHTNESS:
        result = set_brightness(value);
        break;
    case ARDUCAM_SETTING_CONTRAST:
        result =  set_contrast(value);
        break;
    case ARDUCAM_SETTING_SATURATION:
        result = set_saturation(value);
        break;
    case ARDUCAM_SETTING_GAINCEILING:
        result = set_gainceiling(value);
        break;
    case ARDUCAM_SETTING_MIRROR:
        result = set_mirror(value);
        break;
    case ARDUCAM_SETTING_FLIP:
        result = set_flip(value);
        break;
    case ARDUCAM_SETTING_SPECIALEFFECT:
        result = set_specialeffect(value);
        break;
    default:
        result = SL_STATUS_INVALID_PARAMETER;
        break;
    }

    return result;
}

/*************************************************************************************************/
sl_status_t ov2640_get_setting(arducam_setting_t setting, int32_t *value_ptr)
{
    return SL_STATUS_NOT_SUPPORTED;
}



/*************************************************************************************************/
static sl_status_t check_comms_interface(void)
{
    ov2640_id_t id;

    OV2640_WRITE_VERIFY(BANK_SEL, BANK_SEL_SENSOR);

    arducam_driver_i2c_read_reg(MIDH, &id.Manufacturer_ID1);
    arducam_driver_i2c_read_reg(MIDL, &id.Manufacturer_ID2 );
    arducam_driver_i2c_read_reg(REG_PID, &id.PIDH);
    arducam_driver_i2c_read_reg(REG_VER, &id.PIDL);

    if(!(id.PIDH == OV2640_ID))
    {
        return SL_STATUS_NOT_SUPPORTED;
    }

    return SL_STATUS_OK;
}



/*************************************************************************************************/
static sl_status_t set_format(int format)
{
    const reg_addr_value_t *regs;


    switch((arducam_data_format_t)format)
    {
    case ARDUCAM_DATA_FORMAT_RGB888:
    case ARDUCAM_DATA_FORMAT_RGB565:
    case ARDUCAM_DATA_FORMAT_GRAYSCALE:
        regs = ov2640_rgb565_regs;
        break;

    case ARDUCAM_DATA_FORMAT_YUV422:
        regs = ov2640_yuv422_regs;
        break;

    default:
        return SL_STATUS_INVALID_PARAMETER;
    }

    SL_VERIFY(arducam_driver_i2c_write_regs(regs, NULL, 0));

    return SL_STATUS_OK;
}

/*************************************************************************************************/
static sl_status_t set_framesize(arducam_resolution_t resolution, uint32_t width, uint32_t height)
{
    const reg_addr_value_t *regs;


    if((width == 0) || (width % 4) != 0 || (height == 0) || (height % 4) != 0)
    {
        return SL_STATUS_INVALID_PARAMETER;
    }


    switch(resolution)
    {
    case ARDUCAM_RESOLUTION_160x120:
    case ARDUCAM_RESOLUTION_176x144:
    case ARDUCAM_RESOLUTION_320x240:
    case ARDUCAM_RESOLUTION_352x288:
        regs = ov2640_cif;
        break;

    case ARDUCAM_RESOLUTION_640x480:
    case ARDUCAM_RESOLUTION_800x600:
        regs = ov2640_svga;
        break;

    case ARDUCAM_RESOLUTION_1024x768:
    case ARDUCAM_RESOLUTION_1280x1024:
    case ARDUCAM_RESOLUTION_1600x1200:
        regs = ov2640_uxga;
        break;

    case ARDUCAM_RESOLUTION_AUTO:
    {
        if(height <= 288)
        {
            regs = ov2640_cif;
        }
        else if(height <= 600)
        {
            regs = ov2640_svga;
        }
        else 
        {
            regs = ov2640_uxga;
        }
    } break;

    default:
        return SL_STATUS_INVALID_PARAMETER;
    }

    SL_VERIFY(arducam_driver_i2c_write_regs(regs, NULL, 0));

    SL_VERIFY(arducam_driver_i2c_write_reg(ZMOW, ZMOW_OUTW_SET(width)));
    SL_VERIFY(arducam_driver_i2c_write_reg(ZMOH, ZMOH_OUTH_SET(height)));
    SL_VERIFY(arducam_driver_i2c_write_reg(ZMHH, ZMHH_OUTW_SET(width) | ZMHH_OUTH_SET(height)));
    SL_VERIFY(arducam_driver_i2c_write_reg(RESET, 0x00));

    return SL_STATUS_OK;
}

/*************************************************************************************************/
static sl_status_t set_contrast(int level)
{
    const int8_t val = (int8_t)(level + (ARDUCAM_CONTRAST_LEVELS / 2 + 1));
    if (val < 0 || val > ARDUCAM_CONTRAST_LEVELS)
    {
        return SL_STATUS_INVALID_PARAMETER;
    }

    /* Switch to DSP register bank */
    OV2640_WRITE_VERIFY(BANK_SEL, BANK_SEL_DSP);

    /* Write contrast registers */
    for (int i=0; i< ARRAY_COUNT(ov2640_contrast_regs[0]); i++)
    {
         OV2640_WRITE_VERIFY(ov2640_contrast_regs[0][i], ov2640_contrast_regs[val][i]);
    }

    return SL_STATUS_OK;
}

/*************************************************************************************************/
static sl_status_t set_saturation(int level)
{
    const int8_t val = (int8_t)(level + (ARDUCAM_SATURATION_LEVELS / 2 + 1));
    if (val < 0 || val > ARDUCAM_SATURATION_LEVELS)
    {
        return SL_STATUS_INVALID_PARAMETER;
    }

    /* Switch to DSP register bank */
    OV2640_WRITE_VERIFY(BANK_SEL, BANK_SEL_DSP);

    /* Write contrast registers */
    for (int i=0; i< ARRAY_COUNT(ov2640_saturation_regs[0]); i++)
    {
         OV2640_WRITE_VERIFY(ov2640_saturation_regs[0][i], ov2640_saturation_regs[val][i]);
    }

    return SL_STATUS_OK;
}

/*************************************************************************************************/
static sl_status_t set_brightness(int level)
{
    const int8_t val = (int8_t)(level + (ARDUCAM_BRIGHTNESS_LEVELS / 2 + 1));
    if (val < 0 || val > ARDUCAM_BRIGHTNESS_LEVELS)
    {
        return SL_STATUS_INVALID_PARAMETER;
    }

    /* Switch to DSP register bank */
    OV2640_WRITE_VERIFY(BANK_SEL, BANK_SEL_DSP);

    /* Write contrast registers */
    for (int i=0; i< ARRAY_COUNT(ov2640_brightness_regs[0]); i++)
    {
         OV2640_WRITE_VERIFY(ov2640_brightness_regs[0][i], ov2640_brightness_regs[val][i]);
    }

    return SL_STATUS_OK;
}



/*************************************************************************************************/
static sl_status_t set_gainceiling(int gainceiling)
{
    if(gainceiling < ARDUCAM_GAINCEILING_MIN || gainceiling > ARDUCAM_GAINCEILING_MAX)
    {
        return SL_STATUS_INVALID_PARAMETER;
    }
    /* Switch to SENSOR register bank */
    OV2640_WRITE_VERIFY(BANK_SEL, BANK_SEL_SENSOR);

    /* Write gain ceiling register */
    OV2640_WRITE_VERIFY(COM9, COM9_AGC_SET(gainceiling));

    return SL_STATUS_OK;
}

/*************************************************************************************************/
static sl_status_t set_flip(int flip)
{
    uint8_t reg04;
    OV2640_WRITE_VERIFY(BANK_SEL, BANK_SEL_SENSOR);
    SL_VERIFY(arducam_driver_i2c_read_reg( 0x04, &reg04));
    if( flip == 0 )
    {
        reg04 &= 0xAF;
    }
    else
    {
        reg04 |= 0x50;
    }

    OV2640_WRITE_VERIFY(0x04,reg04);

    return SL_STATUS_OK;
}

/*************************************************************************************************/
static sl_status_t set_mirror(int mirror)
{
    uint8_t reg04;

    OV2640_WRITE_VERIFY(BANK_SEL, BANK_SEL_SENSOR);
    SL_VERIFY(arducam_driver_i2c_read_reg( 0x04, &reg04));

    if( mirror == 0 )
    {
        reg04 &= 0x7F;
    }
    else
    {
        reg04 |= 0x80;
    }

    OV2640_WRITE_VERIFY(0x04,reg04);

    return SL_STATUS_OK;
}

/*************************************************************************************************/
static sl_status_t set_specialeffect(int effect)
{
    if(effect < ARDUCAM_SPECIALEFFECT_NONE || effect >= ARDUCAM_SPECIALEFFECT_COUNT)
    {
        return SL_STATUS_INVALID_PARAMETER;
    }

    /* Switch to DSP register bank */
    OV2640_WRITE_VERIFY(BANK_SEL, BANK_SEL_DSP);

    /* Write contrast registers */
    for (int i=0; i< ARRAY_COUNT(ov2640_special_effect_regs[0]); i++)
    {
         OV2640_WRITE_VERIFY(ov2640_special_effect_regs[0][i], ov2640_special_effect_regs[effect+1][i]);
    }

    return SL_STATUS_OK;
}
