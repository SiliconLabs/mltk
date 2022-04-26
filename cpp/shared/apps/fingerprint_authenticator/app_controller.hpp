#pragma once 


#include "logging/logging.hpp"
#include "fingerprint_reader/fingerprint_reader.h"
#include "sl_simple_button_instances.h"


namespace mltk 
{

class AppController 
{
public:

    enum class State
    {
        init,
        idle,
        reading,
        readError,
        signatureInvalid,
        placeFinger,
        displayCurrentUser,
        eraseSignaturesInit,
        erasedSignatures,
        fatalError,
    };


    enum class Mode
    {
        normal,
        saveFingerprints,
        iterateUsers,
        eraseSignatures
    };


    AppController() = default;

    bool init();
    bool process();

    State current_state() const;
    Mode current_mode() const;
    int32_t current_user_id();
    void update_state(State state, bool wait = true);
    void update_mode(Mode mode);
    bool display_user(int32_t user_id, bool check_button=false);
    uint32_t current_timestamp();
    void delay_ms(uint32_t ms);
    int32_t increment_to_next_user_id();
    bool ensure_should_erase_user_signatures(bool &should_erase);
    bool button_is_pressed(
        const sl_button_t* handle, 
        uint32_t& active_timestamp, 
        unsigned min_time_ms,
        unsigned max_time_ms,
        uint32_t* elapsed_ms = nullptr
    );
    bool button_is_active(const sl_button_t* handle);

    fingerprint_reader_image_t image_buffer;

private:
    State _current_state = State::init;
    Mode _current_mode = Mode::normal;
    int32_t _current_user_id = 0;
    uint32_t _button0_press_context = 0;
    uint32_t _button1_click_context = 0;
    uint32_t _button1_press_context = 0;
};

const char* to_str(AppController::State state);
const char* to_str(AppController::Mode mode);

} // namespace mltk 