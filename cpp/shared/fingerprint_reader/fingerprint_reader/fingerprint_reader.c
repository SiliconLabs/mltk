#include <stdlib.h>
#include <stdint.h>

#include "fingerprint_reader_internal.h"


#include "em_cmu.h"
#include "em_gpio.h"
#include "em_usart.h"
#include "gpiointerrupt.h"


#include "r503_driver.h"


#define CHECK_INITIALIZED() if(!reader_context.initialized) return SL_STATUS_NOT_INITIALIZED


typedef struct
{
    void (*finger_detected_irq_callback)(void);
    volatile bool finger_detected;
    bool initialized;
} fingerprint_reader_t;




static void on_finger_detected_irq_handler(uint8_t irq_no);





static fingerprint_reader_t reader_context;




/*************************************************************************************************/
sl_status_t fingerprint_reader_init(const fingerprint_reader_config_t* config)
{
    sl_status_t status;

    reader_context.initialized = false;

    CMU_ClockEnable(cmuClock_GPIO, true);
#if defined(_CMU_HFPERCLKEN0_MASK)
    CMU_ClockEnable(cmuClock_HFPER, true);
#endif

    // ---------------------------------------------------
    // Initialize USART interface
    // ---------------------------------------------------
    CMU_ClockEnable(USART_CLOCK(FINGERPRINT_READER_USART_PERIPHERAL_NO), true);

    // Configure GPIO mode
    GPIO_PinModeSet(FINGERPRINT_READER_USART_TX_PORT,  FINGERPRINT_READER_USART_TX_PIN,  gpioModePushPull, 1);
    GPIO_PinModeSet(FINGERPRINT_READER_USART_RX_PORT,  FINGERPRINT_READER_USART_RX_PIN,  gpioModeInput, 1); 

	USART_InitAsync_TypeDef usart_config = USART_INITASYNC_DEFAULT;
    usart_config.enable       = usartDisable;

	// Init USART
	USART_InitAsync(FINGERPRINT_READER_USART, &usart_config);

    // Set USART pin locations
#if defined(_SILICON_LABS_32B_SERIES_2)
    GPIO->USARTROUTE[FINGERPRINT_READER_USART_PERIPHERAL_NO].ROUTEEN = GPIO_USART_ROUTEEN_TXPEN
                                           | GPIO_USART_ROUTEEN_RXPEN
                                           | GPIO_USART_ROUTEEN_CLKPEN;

    GPIO->USARTROUTE[FINGERPRINT_READER_USART_PERIPHERAL_NO].TXROUTE = ((uint32_t)FINGERPRINT_READER_USART_TX_PORT
                                            << _GPIO_USART_TXROUTE_PORT_SHIFT)
                                           | ((uint32_t)FINGERPRINT_READER_USART_TX_PIN
                                              << _GPIO_USART_TXROUTE_PIN_SHIFT);

    GPIO->USARTROUTE[FINGERPRINT_READER_USART_PERIPHERAL_NO].RXROUTE = ((uint32_t)FINGERPRINT_READER_USART_RX_PORT
                                            << _GPIO_USART_RXROUTE_PORT_SHIFT)
                                           | ((uint32_t)FINGERPRINT_READER_USART_RX_PIN
                                              << _GPIO_USART_RXROUTE_PIN_SHIFT);

#else
    FINGERPRINT_READER_USART->ROUTELOC0 = (FINGERPRINT_READER_USART_TX_LOC|FINGERPRINT_READER_USART_RX_LOC); 
    FINGERPRINT_READER_USART->ROUTEPEN = USART_ROUTEPEN_TXPEN | USART_ROUTEPEN_RXPEN;
#endif 

    // Enable USART
    USART_Enable(FINGERPRINT_READER_USART, usartEnable);

    status = r503_init();
    if(status != SL_STATUS_OK)
    {
        fingerprint_reader_deinit();
        return status;
    }

    // Enable the external interrupt pin to wake the device
    // when the user places their finger on the reader
    reader_context.finger_detected_irq_callback = config->finger_detected_irq_callback;
    GPIO_PinModeSet(FINGERPRINT_READER_ACTIVITY_PORT, FINGERPRINT_READER_ACTIVITY_PIN, gpioModeInput, 1);
    GPIO_ExtIntConfig(
        FINGERPRINT_READER_ACTIVITY_PORT,
        FINGERPRINT_READER_ACTIVITY_PIN,
        FINGERPRINT_READER_ACTIVITY_PIN,
        false,
        true,
        true
    );
    GPIOINT_Init();
    GPIOINT_CallbackRegister(FINGERPRINT_READER_ACTIVITY_PIN, on_finger_detected_irq_handler);
   
    return SL_STATUS_OK;
}

/*************************************************************************************************/
sl_status_t fingerprint_reader_deinit()
{
    r503_deinit();
    GPIOINT_CallbackUnRegister(FINGERPRINT_READER_ACTIVITY_PIN);
    USART_Reset(FINGERPRINT_READER_USART);
    CMU_ClockEnable(USART_CLOCK(FINGERPRINT_READER_USART_PERIPHERAL_NO), false);
    return SL_STATUS_OK;
}

/*************************************************************************************************/
sl_status_t fingerprint_reader_update_led(const fingerprint_reader_led_config_t* config)
{
    CHECK_INITIALIZED();

    r503_led_config_t led_config;
    led_config.mode = config->mode;
    led_config.color = config->color;
    led_config.count = config->count;
    led_config.speed = config->speed;
    return r503_update_led(&led_config);
}

/*************************************************************************************************/
bool fingerprint_reader_is_image_available()
{
    return reader_context.finger_detected;
}

/*************************************************************************************************/
sl_status_t fingerprint_reader_get_image(fingerprint_reader_image_t image_buffer)
{
    CHECK_INITIALIZED();
    
    sl_status_t status;
    r503_status_t command_status;
    
    for(int i = 0; i < 10; ++i)
    {
        status = r503_capture_image(&command_status);
        if(status != SL_STATUS_OK)
        {
            break;
        }
        else if(command_status == R503_STATUS_OK)
        {
            status = r503_read_image(image_buffer);
            break;
        }
        else if(command_status == R503_STATUS_NO_FINGER)
        {
            status = SL_STATUS_EMPTY;
        }
        else if(command_status == R503_STATUS_BAD_IMAGE_QUALITY)
        {
            status = SL_STATUS_INVALID_STATE;
        }
        else if(command_status != R503_STATUS_OK)
        {
            status = SL_STATUS_FAIL;
        }
    }

    // The reader will assert the wakeup signal when invoking the CAPTURE command
    // So we must clear the flag here as opposed to the beginning of this function
    reader_context.finger_detected = false;

    return status;
}





/*************************************************************************************************/
static void on_finger_detected_irq_handler(uint8_t irq_no)
{
    reader_context.finger_detected = true;
    if(reader_context.finger_detected_irq_callback != NULL)
    {
        reader_context.finger_detected_irq_callback();
    }
}