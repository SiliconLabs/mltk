#include "app_controller.hpp"
#include "app_config.hpp"
#include "sl_sleeptimer.h"
#include  "mltk_tflite_micro_helper.hpp"
#include "fingerprint_reader/fingerprint_reader.h"
#include "fingerprint_vault.h"


namespace mltk
{


struct LedConfig
{
    fingerprint_reader_led_config_t config;
    uint8_t duration_seconds;
};
#define ADD_LED_CONFIG(color, mode, speed, duration) \
{ \
    { \
        fpr ## color ## Led, \
        fpr ## mode ## Mode, \
        speed, \
        0, \
    }, \
    duration \
}

static const LedConfig LED_STATES[] = 
{
    //             Color     Mode    Speed Duration(seconds)
    ADD_LED_CONFIG(Blue,   Breathing, 2,    0), // init
    ADD_LED_CONFIG(Blue,   Breathing, 100,  0), // idle
    ADD_LED_CONFIG(Purple, Breathing, 50,   0), // reading
    ADD_LED_CONFIG(Red,    Flashing,  75,   3), // readError
    ADD_LED_CONFIG(Purple, Flashing,  25,   3), // signatureInvalid
    ADD_LED_CONFIG(Blue,   Flashing,  200,  2), // placeFinger
    ADD_LED_CONFIG(Purple, AlwaysOn,  200,  3), // displayCurrentUser
    ADD_LED_CONFIG(Purple, Breathing, 100,  0), // eraseSignaturesInit
    ADD_LED_CONFIG(Purple, Flashing,  50,   5), // erasedSignatures
    ADD_LED_CONFIG(Red,    Flashing,  15,   3), // fatalError
};


/*************************************************************************************************/
bool AppController::init()
{
    return true;
}

/*************************************************************************************************/
bool AppController::process()
{
    bool retval = false;
    uint32_t elapsed_ms;

    if(fingerprint_reader_is_image_available())
    {
        return true;
    }

    delay_ms(10); // 10ms for debounce

    if(button_is_pressed(&sl_button_btn0, _button0_press_context,  75, 1000))
    {
        MLTK_DEBUG("Button 1 clicked");
        update_mode(Mode::saveFingerprints);
        retval = true;
    }
    else if(button_is_pressed(&sl_button_btn1, _button1_click_context, 75, 1000))
    {
        MLTK_DEBUG("Button 2 clicked");
        update_mode(Mode::iterateUsers);
        retval = true;
    }
    else if(button_is_pressed(&sl_button_btn1, _button1_press_context, 7000, 10000, &elapsed_ms) || elapsed_ms > 1500)
    {
        if(elapsed_ms > 1500)
        {
            update_mode(Mode::eraseSignatures);
            MLTK_DEBUG("Button 2 pressed");
            retval = true;
        }
    }

    if(retval)
    {
        _button0_press_context = 0;
        _button1_click_context = 0;
        _button1_press_context = 0;
    }
    else if(_button0_press_context != 0 || 
       _button1_click_context != 0 || 
       _button1_press_context != 0)
    {
        retval = true;
    }

    return retval;
}


/*************************************************************************************************/
AppController::State AppController::current_state() const
{
    return _current_state;
}


/*************************************************************************************************/
AppController::Mode AppController::current_mode() const
{
    return _current_mode;
}


/*************************************************************************************************/
int32_t AppController::current_user_id()
{
    return _current_user_id;
}


/*************************************************************************************************/
void AppController::update_mode(Mode mode)
{
    if(_current_mode != mode)
    {
        MLTK_DEBUG("New mode: %s -> %s", to_str(_current_mode), to_str(mode));
        _current_mode = mode;
    }
}

/*************************************************************************************************/
void AppController::update_state(State state, bool wait)
{
    if(_current_state == State::fatalError)
    {
        return;
    }

    if(_current_state != state)
    {
        MLTK_DEBUG("New state: %s -> %s", to_str(_current_state), to_str(state));
       
        for(int i = 10; i > 0; ++i)
        {
            if(fingerprint_reader_update_led(&LED_STATES[(int)state].config) == SL_STATUS_OK)
            {
                _current_state = state;
                break;
            }
        }
    }

    if(wait)
    {
        const uint32_t ms = LED_STATES[(int)state].duration_seconds * 1000;
        delay_ms(ms);
    }
}   


/*************************************************************************************************/
bool AppController::display_user(int32_t user_id, bool check_button)
{
    fingerprint_reader_led_config_t led_config; 

    _current_state = AppController::State::displayCurrentUser;

    led_config.color = (user_id == 0) ? fprRedLed : (user_id == 1) ? fprBlueLed : fprPurpleLed;
    led_config.mode = fprAlwaysOnMode;
    led_config.count = 0;
    led_config.speed = 0;
    fingerprint_reader_update_led(&led_config);

    if(check_button)
    {
        int signature_count = fingerprint_vault_count_user_signatures(user_id);
        MLTK_INFO("  %d saved signatures", signature_count);

        const uint32_t start_time = current_timestamp();
        while((current_timestamp() - start_time) < 2500)
        {
            if(button_is_pressed(&sl_button_btn1, _button1_click_context,  75, 1000))
            {
                MLTK_DEBUG("Button 2 clicked");
                return true;
            }
        }
    }
    else 
    {
        delay_ms(2500);
    }

    return false;
}



/*************************************************************************************************/
bool AppController::ensure_should_erase_user_signatures(bool &should_erase)
{
    uint32_t elapsed_time;
    should_erase = false;

    if(button_is_pressed(&sl_button_btn1, _button1_press_context, 7000, 20000, &elapsed_time) || elapsed_time > 10000)
    {
        should_erase = true;
    }

    return _button1_press_context > 0;
}


/*************************************************************************************************/
uint32_t AppController::current_timestamp()
{
    return sl_sleeptimer_tick_to_ms(sl_sleeptimer_get_tick_count());
}


/*************************************************************************************************/
void AppController::delay_ms(uint32_t ms)
{
    const uint32_t start_time = current_timestamp();
    while((current_timestamp() - start_time) < ms)
    {
    }
}

/*************************************************************************************************/
int32_t AppController::increment_to_next_user_id()
{
    _current_user_id = (_current_user_id + 1) % MAX_USERS;
    return _current_user_id;
}



/*************************************************************************************************/
bool AppController::button_is_pressed(
    const sl_button_t* handle, 
    uint32_t& active_timestamp, 
    unsigned min_time_ms, 
    unsigned max_time_ms,
    uint32_t* elapsed_ms
)
{
    const bool is_active = button_is_active(handle);
    
    if(elapsed_ms != nullptr)
    {
        *elapsed_ms = 0;
    }

    if(active_timestamp == 0 && is_active)
    {
        active_timestamp = current_timestamp();
    }
    else if(active_timestamp != 0)
    {
        const uint32_t elapsed_time = current_timestamp() - active_timestamp;
        if(elapsed_ms != nullptr)
        {
            *elapsed_ms = elapsed_time;
        }

        if(!is_active)
        {
            active_timestamp = 0;
 
            if(elapsed_time >= min_time_ms && elapsed_time <= max_time_ms)
            {
                MLTK_DEBUG("elapsed: %d", elapsed_time);
                return true;
            }
        }
    }

    return false;
}

/*************************************************************************************************/
bool AppController::button_is_active(const sl_button_t* handle)
{
    return sl_button_get_state(handle) == SL_SIMPLE_BUTTON_PRESSED;
}


/*************************************************************************************************/
const char* to_str(AppController::State state)
{
    switch(state)
    {
    case AppController::State::init: return "init";
    case AppController::State::idle: return "idle";
    case AppController::State::reading: return "reading";
    case AppController::State::readError: return "readError";
    case AppController::State::signatureInvalid: return "signatureInvalid";
    case AppController::State::placeFinger: return "placeFinger";
    case AppController::State::displayCurrentUser: return "displayCurrentUser";
    case AppController::State::eraseSignaturesInit: return "eraseSignaturesInit";
    case AppController::State::erasedSignatures: return "erasedSignatures";
    case AppController::State::fatalError: return "fatalError";
    default: return "unknown";
    }
}


/*************************************************************************************************/
const char* to_str(AppController::Mode mode)
{
    switch(mode)
    {
        case AppController::Mode::normal: return "normal";
        case AppController::Mode::saveFingerprints: return "saveFingerprints";
        case AppController::Mode::iterateUsers: return "iterateUsers";
        case AppController::Mode::eraseSignatures: return "eraseSignatures";
        default: return "unknown";
    }
}


} // namespace mltk