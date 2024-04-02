
#include "em_emu.h"

#include "sl_system_init.h"
#include "logging/logging.hpp"
#include "jlink_stream/jlink_stream.hpp"
#include "jlink_stream/jlink_stream_internal.hpp"
#include "fingerprint_reader/fingerprint_reader.h"

#include "tflite_micro_model/tflite_micro_utils.hpp"


#include "app_controller.hpp"
#include "fingerprint_authenticator.hpp"
#include "mltk_fingerprint_authenticator_model_generated.hpp"

#include "app_config.hpp"


using namespace mltk;


// This is defined by the GSDK NVM library
extern "C" const uint32_t linker_nvm_begin;


AppController app_controller;
FingerprintAuthenticator fingerprint_authenticator;


static void process_normal_mode();
static void process_save_fingerprint_mode();
static void process_iterate_users_mode();
static void process_erase_signature_mode();



/*************************************************************************************************/
extern "C" int main(void)
{
    sl_status_t status;
    const uint8_t* model_flatbuffer;

    sl_system_init();


    auto& logger = get_logger();

    MLTK_INFO("Fingerprint Authenticator app starting\n");

    // First check if a new .tflite was programmed to the end of flash
    // (This will happen when this app is executed from the command-line: "mltk classify_image my_model")
    if(!mltk::TfliteMicroModelHelper::get_tflite_flatbuffer_from_end_of_flash(&model_flatbuffer, nullptr, &linker_nvm_begin))
    {
        // If no .tflite was programmed, then just use the default model
        printf("Using default model built into application\n");
        model_flatbuffer = mltk_model_flatbuffer;
    }



    // This is used to transfer fingerprint images to the command:
    // mltk fingerprint_reader
    jlink_stream::register_stream("raw", jlink_stream::Write);
    jlink_stream::register_stream("proc", jlink_stream::Write);


    if(!app_controller.init())
    {
        MLTK_ERROR("Failed to initialize app_controller");
        app_controller.update_state(AppController::State::fatalError);
        return -1;
    }
    if(!fingerprint_authenticator.init())
    {
        MLTK_ERROR("Failed to initialize fingerprint_authenticator");
        app_controller.update_state(AppController::State::fatalError);
        return -1;
    }

    if(!mltk::TfliteMicroModelHelper::verify_model_flatbuffer(mltk_model_flatbuffer, mltk_model_flatbuffer_length))
    {
        MLTK_ERROR("Invalid model flatbuffer");
        app_controller.update_state(AppController::State::fatalError);
        return -1;
    }
    
    if(!fingerprint_authenticator.load_model(model_flatbuffer))
    {
        MLTK_ERROR("Failed to load model");
        app_controller.update_state(AppController::State::fatalError);
        return -1;
    }

    // Dump a summary of the model
    logger.info("--------------------------------------");
    logger.info("Model information:");
    fingerprint_authenticator.model.print_summary();


    // Initialize the fingerprint reader
    fingerprint_reader_config_t fpr_config = FINGERPRINT_READER_DEFAULT_CONFIG;
    status = fingerprint_reader_init(&fpr_config);
    if(status != SL_STATUS_OK)
    {
        MLTK_ERROR("Failed to initialize the finerprint reader, err: %u", status);
        app_controller.update_state(AppController::State::fatalError);
        return -1;
    }


    logger.info("\n--------------------------------------");
    logger.info("Application Instructions:\n");

    logger.info("* Click button 2 to iterate through users");
    logger.info("  The LED will be solid red, blue, or purple to signify the current user:");
    logger.info("  - RED    -> user 0");
    logger.info("  - BLUE   -> user 1");
    logger.info("  - PURPLE -> user 2\n");

    logger.info("* Press button 2 for 10s then release to erase the current user's signatures.");
    logger.info("  The LED will pulse purple while the erase sequence initializes,");
    logger.info("  and flash purple when the signatures are erased.");
    logger.info("  Release button 2 before 10s have elapsed to abort the sequence.\n");

    logger.info("* Click button 1 to save the fingerprints for the current user.");
    logger.info("  The LED will flash blue when you should place your finger on the reader.");
    logger.info("  If the LED flashes red then there was a reading error,");
    logger.info("  wait for the LED to flash blue to try again.");
    logger.info("  This sequence will repeat %d times.", SAVED_FINGERPRINT_COUNT);
    logger.info("  i.e. The SAME finger will be captured %d times.", SAVED_FINGERPRINT_COUNT);
    logger.info("  If there is no activity on the reader after 7s,");
    logger.info("  then this sequence will be aborted.\n");

    logger.info("* In normal operation, the LED pulses blue.");
    logger.info("  Place your finger on the reader to authenticate.");
    logger.info("  The LED will pulse purple while your finger is processed.");
    logger.info("  The LED will be solid red, blue, or purple for the authenticated user.");
    logger.info("  The LED will flash purple for an unknown fingerprint\n");

    logger.info("\n\nHINT: Run the following command to view the captured fingerprints:\n");
    logger.info("mltk fingerprint_reader");

    
    if(fingerprint_authenticator.is_disabled())
    {
        MLTK_WARN("Inference disabled, not generating signature from fingerprint");
    }

    MLTK_DEBUG("\n\nApp loop starting ...");
    for(;;)
    {
        // Break out of the loop if a fatal error occurred
        if(app_controller.current_state() == AppController::State::fatalError)
        {
            break;
        }

        app_controller.update_state(AppController::State::idle);
       
        // If we are not actively processing
        if(!app_controller.process())
        {        
            // Then wait for either a button to be pressed
            // OR for fingerprint reader activity
            //MLTK_DEBUG("sleep");
            EMU_EnterEM1();
            continue;

        }

        // Process based on the current app mode
        switch(app_controller.current_mode())
        {
        case AppController::Mode::normal:
            process_normal_mode();
            break;
        case AppController::Mode::saveFingerprints:
            process_save_fingerprint_mode();
            break;
        case AppController::Mode::iterateUsers:
            process_iterate_users_mode();
            break;
        case AppController::Mode::eraseSignatures:
            process_erase_signature_mode();
            break;
        default:
            break;
        }

        app_controller.update_mode(AppController::Mode::normal);
    }

    return -1;
}


/*************************************************************************************************/
static void process_normal_mode()
{
    FingerprintSignature signature;
    int32_t user_id;


    // Check if a fingerprint is ready to be read
    if(!fingerprint_reader_is_image_available())
    {
        // No fingerprint is available to be read
        // so just return
        return;
    }

    MLTK_INFO("Reading fingerprint from reader");

    // Notify the user that the reader is actively be read
    app_controller.update_state(AppController::State::reading, false);

    // Read the fingerprint from the sensor
    jlink_stream_set_interrupt_enabled(false);
    sl_status_t status = fingerprint_reader_get_image(app_controller.image_buffer);
    jlink_stream_set_interrupt_enabled(true);
    if(status != SL_STATUS_OK)
    {
        if(status == SL_STATUS_EMPTY)
        {
            MLTK_WARN("No finger detected, please try again");
            app_controller.update_state(AppController::State::readError);
        }
        else if(status == SL_STATUS_INVALID_STATE)
        {
            MLTK_WARN("Bad fingerprint quality, please try again");
            app_controller.update_state(AppController::State::readError);
        }
        else 
        {
            MLTK_WARN("Failed to read sensor, err: %d", status);
            app_controller.update_state(AppController::State::fatalError, false);
        }
        return;
    }

    // Dump the raw fingerprint image if the Python script is connected
    jlink_stream::write_all("raw", app_controller.image_buffer, sizeof(fingerprint_reader_image_t));

    if(fingerprint_authenticator.is_disabled())
    {
        return;
    }


    MLTK_INFO("Generating signature from fingerprint");

    // Use ML to generate a signature from the fingerprint image
    bool fingerprint_image_valid;
    if(!fingerprint_authenticator.generate_signature(app_controller.image_buffer, signature, fingerprint_image_valid))
    {
        MLTK_WARN("Error while generating fingerprint signature");
        app_controller.update_state(AppController::State::fatalError);
        return;
    }
    else if(!fingerprint_image_valid)
    {
        MLTK_WARN("The captured fingerprint has poor quality, please try again");
        app_controller.update_state(AppController::State::readError);
        return;
    }

    MLTK_INFO("Authenticating signature: %s", FingerprintAuthenticator::signature_to_str(signature));

    // Authenticate the signature
    if(!fingerprint_authenticator.authenticate_signature(signature, user_id))
    {
        MLTK_WARN("Failed to authenticate signature");
        app_controller.update_state(AppController::State::fatalError, false);
        return;
    }

    if(user_id == -1)
    {
        MLTK_INFO("Unknown fingerprint");
        app_controller.update_state(AppController::State::signatureInvalid);
    }
    else 
    {
        MLTK_INFO("Fingerprint belongs to user_id: %d", user_id);
        app_controller.display_user(user_id);
    }
}


/*************************************************************************************************/
static void process_save_fingerprint_mode()
{
    const int32_t current_user_id = app_controller.current_user_id();

    MLTK_INFO("Collecting %d fingerprints for user %d", SAVED_FINGERPRINT_COUNT, current_user_id);

    // Display the current user's LED pattern for a moment
    app_controller.display_user(current_user_id);


    MLTK_INFO("Place finger on reader ...");

    // Collect N fingerprint samples
    for(int sample_count = 0;;)
    {
        FingerprintSignature signature;
        const uint32_t start_time = app_controller.current_timestamp();

        // Notify the user that they should place their finger on the reader
        app_controller.update_state(AppController::State::placeFinger, false);

        // Wait until the user puts their finger on the reader
        for(;;)
        {
            // If too much time has passed without activity,
            // then just return
            if((app_controller.current_timestamp() - start_time) > 10000)
            {
                MLTK_WARN("Timed-out waiting for user to place finger on reader");
                return;
            }

            // Check if a fingerprint is ready to be read
            if(fingerprint_reader_is_image_available())
            {
                // An image is ready,
                // break out of the loop and read the sensors
                break;
            }
        }

        MLTK_INFO("Reading fingerprint from reader");

        // Notify the user that the reader is actively be read
        app_controller.update_state(AppController::State::reading, false);

        // Read the fingerprint from the sensor
        sl_status_t status = fingerprint_reader_get_image(app_controller.image_buffer);
        if(status == SL_STATUS_EMPTY)
        {
            MLTK_WARN("No finger detected, please try again");
            app_controller.update_state(AppController::State::readError);
            continue;
        }
        else if(status != SL_STATUS_OK)
        {
            MLTK_WARN("Failed to read sensor, err: %d", status);
            app_controller.update_state(AppController::State::fatalError, false);
            return;
        }

        // Dump the raw fingerprint image if the Python script is connected
        jlink_stream::write_all("raw", app_controller.image_buffer, sizeof(fingerprint_reader_image_t));

        MLTK_INFO("Generating signature from fingerprint");

        // Use ML to generate a signature from the fingerprint image
        bool fingerprint_image_valid;
        if(!fingerprint_authenticator.generate_signature(app_controller.image_buffer, signature, fingerprint_image_valid))
        {
            MLTK_WARN("Error while generating fingerprint signature");
            app_controller.update_state(AppController::State::fatalError);
            // Continue to the beginning of the loop
            return;
        }
        else if(!fingerprint_image_valid)
        {
            MLTK_WARN("The captured fingerprint has poor quality, ensure your finger is clean and please try again");
            app_controller.update_state(AppController::State::readError);
            // Continue to the beginning of the loop
            continue;
        }

        MLTK_DEBUG("Signature: %s", FingerprintAuthenticator::signature_to_str(signature));
        MLTK_INFO("Saving signature %d of %d", sample_count+1, SAVED_FINGERPRINT_COUNT);

        // Save the signature for the current user
        if(!fingerprint_authenticator.save_signature(signature, current_user_id))
        {
            MLTK_WARN("Failed to save signature for current user");
            app_controller.update_state(AppController::State::fatalError);
            return;
        }

        // If we've collected enough signatures
        // then exit the loop
        sample_count++;
        if(sample_count >= SAVED_FINGERPRINT_COUNT)
        {
            MLTK_INFO("Finished collecting fingerprints");
            break;
        }
        MLTK_INFO("Place your SAME finger on the reader ...");
        app_controller.update_state(AppController::State::placeFinger, false);
    }

}


/*************************************************************************************************/
static void process_iterate_users_mode()
{
    bool button_pressed;

    do
    {
        const int32_t current_user_id = app_controller.increment_to_next_user_id();
        MLTK_INFO("Current user: %d", current_user_id);
        button_pressed = app_controller.display_user(current_user_id, true);
    } while(button_pressed);

}


/*************************************************************************************************/
static void process_erase_signature_mode()
{
     const int32_t current_user_id = app_controller.current_user_id();

    MLTK_INFO("Continue to press button 2 for 10s then release to erase user %d's signatures", current_user_id);
    app_controller.update_state(AppController::State::eraseSignaturesInit, false);
    
    bool button_pressed;
    bool should_erase = false;
    bool printed_msg = false;

    do 
    {
        button_pressed = app_controller.ensure_should_erase_user_signatures(should_erase);
        if(should_erase && !printed_msg)
        {
            printed_msg = true;
            MLTK_INFO("Release button 2 to erase user %d's signatures", current_user_id);
        }
    } while(button_pressed);

    if(!should_erase)
    {
        MLTK_INFO("Aborting erasing user signatures");
        return;
    }


    MLTK_INFO("Erasing signatures for user %d", current_user_id);
    app_controller.update_state(AppController::State::erasedSignatures, false);

    if(!fingerprint_authenticator.remove_signatures(current_user_id))
    {
        MLTK_WARN("Failed to erase user signatures");
        app_controller.update_state(AppController::State::fatalError);
        return;
    }
    app_controller.update_state(AppController::State::erasedSignatures, true);

    MLTK_INFO("Signatures erased");
}
