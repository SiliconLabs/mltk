/***************************************************************************//**
 * @file
 * @brief Top level application functions
 *******************************************************************************
 * # License
 * <b>Copyright 2022 Silicon Laboratories Inc. www.silabs.com</b>
 *******************************************************************************
 *
 * The licensor of this software is Silicon Laboratories Inc. Your use of this
 * software is governed by the terms of Silicon Labs Master Software License
 * Agreement (MSLA) available at
 * www.silabs.com/about-us/legal/master-software-license-agreement. This
 * software is distributed to you in Source Code format and is governed by the
 * sections of the MSLA applicable to Source Code.
 *
 ******************************************************************************/
#include <cmath>
#include <algorithm>
#include <cstdio>

#include "sl_power_manager.h"
#include "sl_status.h"
#include "sl_led.h"
#include "sl_simple_led_instances.h"
#include "sl_sleeptimer.h"
#include "all_ops_resolver.h"
#include "tflite_micro_model/tflite_micro_model.hpp"
#include "tflite_micro_model/tflite_micro_utils.hpp"
#include "mltk_tflite_micro_helper.hpp"
#include "arducam/arducam.h"
#include "jlink_stream/jlink_stream.hpp"

#include "image_classifier.h"
#include "recognize_commands.h"



#define DETECTION_LED sl_led_led1
#define ACTIVITY_LED sl_led_led0


#ifdef SL_CATALOG_KERNEL_PRESENT
#include "os.h"

static OS_TCB tcb;
static CPU_STK stack[TASK_STACK_SIZE];

static void image_classifier_task(void *arg);

#else
static sl_sleeptimer_timer_handle_t inference_timer;
#endif

#if SL_SIMPLE_LED_COUNT < 2
  #error "Sample application requires two leds"
#endif

static tflite::AllOpsResolver op_resolver;
static RecognizeCommands *command_recognizer = nullptr;
static mltk::TfliteMicroModel model;

static int32_t detected_timeout = 0;
static int32_t activity_timestamp = 0;
static int32_t activity_toggle_timestamp = 0;
static uint8_t previous_score = 0;
static int32_t previous_score_timestamp = 0;
static int previous_result = 0;
static mltk::StringList category_labels;
int category_count;

// This is defined by the build scripts
// which converts the specified .tflite to a C array
extern "C" const uint8_t sl_tflite_model_array[];

AppSettings app_settings;


static bool load_model_parameters();
static bool initialize_camera();
static void retrieve_next_camera_image(uint8_t** image_data, uint32_t *image_size);
static void standardize_image_data(uint8_t* image_data, uint32_t image_size);
static void process_inference_output();
static void handle_result(int32_t current_time, int result, uint8_t score, bool is_new_command);
static void dump_image(const uint8_t* image_data, uint32_t image_length);



/************************************************************************//**
 * Initialize audio classifier application.
 ******************************************************************************/
void image_classifier_init(void)
{
    const uint8_t* model_flatbuffer;

    printf("Image Classifier\n");

    // This is used to dump the images to a Python script via JLink stream
    jlink_stream::register_stream("image", jlink_stream::Write);


    // First check if a new .tflite was programmed to the end of flash
    // (This will happen when this app is executed from the command-line: "mltk classify_image my_model")
    if(!mltk::get_tflite_flatbuffer_from_end_of_flash(&model_flatbuffer))
    {
        // If no .tflite was programmed, then just use the default model
        printf("Using default model built into application\n");
        model_flatbuffer = sl_tflite_model_array;
    }

    // Register the accelerator if the TFLM lib was built with one
    mltk::mltk_tflite_micro_register_accelerator();

    // Attempt to load the model using the arena size specified in the .tflite
    if(!model.load(model_flatbuffer, op_resolver))
    {
        printf("ERROR: Failed to load .tflite model\n");
        while(1)
        ;
    }

    model.print_summary();

    // Load the settings embedded into the .tflite model flatbuffer
    if(!load_model_parameters())
    {
        printf("ERROR: Failed to load parameters from .tflite model\n");
        while(1)
        ;
    }

    if(!initialize_camera())
    {
        printf("ERROR: Failed to initialize camera. Is it properly connected?\n");
        while(1)
        ;
    }


  // Instantiate CommandRecognizer
  static RecognizeCommands static_recognizer(
      app_settings.average_window_duration_ms,
      app_settings.detection_threshold,
      app_settings.suppression_count,
      app_settings.minimum_count
);
  command_recognizer = &static_recognizer;

  // Add EM1 requirement to allow microphone sampling
  sl_power_manager_add_em_requirement(SL_POWER_MANAGER_EM1);


#ifdef SL_CATALOG_KERNEL_PRESENT
  RTOS_ERR err;

  // Create Application Task
  char task_name[] = "audio classifier task";
  OSTaskCreate(&tcb,
               task_name,
               image_classifier_task,
               DEF_NULL,
               TASK_PRIORITY,
               &stack[0],
               (TASK_STACK_SIZE / 10u),
               TASK_STACK_SIZE,
               0u,
               0u,
               DEF_NULL,
               (OS_OPT_TASK_STK_CLR),
               &err);

  EFM_ASSERT((RTOS_ERR_CODE_GET(err) == RTOS_ERR_NONE));

#else
  // The device will go to sleep after each loop.
  // This timer will wake it up to execute another loop every app_settings.latency_ms
  sl_status_t status = sl_sleeptimer_start_periodic_timer_ms(
      &inference_timer,
      std::max(app_settings.latency_ms, (uint32_t)1),
      nullptr, nullptr, 0, 0);
  if(status != SL_STATUS_OK)
  {
    printf("ERROR: Failed to start periodic inference timer\n");
    while (1)
      ;
  }
#endif
}

#ifdef SL_CATALOG_KERNEL_PRESENT
/***************************************************************************//**
 * Image classifier task function
 *
 * This function is executed by a Micrium OS task and does not return.
 *
 * @param arg ignored
 ******************************************************************************/
static void image_classifier_task(void *arg)
{
  (void)&arg;
  while(1) {
    RTOS_ERR err;

    app_process_action();

    // Delay task in order to do periodic inference
    OSTimeDlyHMSM(0, 0, 0, std::max(app_settings.latency_ms, (uint32_t)1), OS_OPT_TIME_PERIODIC, &err);
    EFM_ASSERT((RTOS_ERR_CODE_GET(err) == RTOS_ERR_NONE));
  }
}
#endif

/***************************************************************************//**
 * Run a single application loop
 *
 * This function is executed by either the RTOS task or main.c loop
 ******************************************************************************/
void app_process_action()
{
    static uint32_t prev_loop_timestamp = 0;

    uint32_t current_timestamp = sl_sleeptimer_tick_to_ms(sl_sleeptimer_get_tick_count());

    if((current_timestamp - prev_loop_timestamp) < app_settings.latency_ms)
    {
        return;
    }

    // Perform a detection
    prev_loop_timestamp = current_timestamp;

    sl_status_t status;
    uint8_t* image_data;
    uint32_t image_size;

    // Retrieve the image from the camera
    // This will also dump the image to the JLink stream if necessary
    retrieve_next_camera_image(&image_data, &image_size);

    // Standardize the image data (if necessary)
    // and copy to model input tensor
    standardize_image_data(image_data, image_size);

    // Release the image now that it has been copied to the input tensor
    arducam_release_image();

    if(app_settings.enable_inference)
    {
        // Run the model inference
        if(!model.invoke())
        {
            printf("Failed to run inference\n");
            while (1)
            ;
        }

        // Process the inference results
        process_inference_output();
    }
}

/***************************************************************************//**
 * Load the parameters embedded into the .tflite model file
 ******************************************************************************/
static bool load_model_parameters()
{
    if(!model.parameters.get("classes", category_labels))
    {
      printf("ERROR: Model does not contain a 'classes' parameter\n");
      return false;
    }
    category_count = category_labels.size();

    const auto input_tensor = model.input();


    // Attempt to retrieve the model parameters from the .tflite flatbuffer
    model.parameters.get("samplewise_norm.rescale", app_settings.samplewise_norm_rescale);
    if(app_settings.samplewise_norm_rescale != 0)
    {
        printf("Image data scaler = %f\n", app_settings.samplewise_norm_rescale);
    }

    model.parameters.get("samplewise_norm.mean_and_std", app_settings.samplewise_norm_mean_and_std);
    if(app_settings.samplewise_norm_mean_and_std)
    {
        printf("Using samplewise mean & STD normalization\n");
    }

    if(app_settings.samplewise_norm_rescale != 0 || app_settings.samplewise_norm_mean_and_std)
    {
        if(input_tensor->type != kTfLiteFloat32)
        {
           printf(
                "ERROR: If using image scaling or samplewise mean/STD normalization, then the model input type must be float32\n"
            );
            return false;
        }
    }
    else if(!(input_tensor->type == kTfLiteInt8 || input_tensor->type == kTfLiteFloat32))
    {
        printf("ERROR: Model data type must either be int8 or float32\n");
        return false;
    }


    model.parameters.get("verbose_inference_output", app_settings.verbose_inference_output);
    model.parameters.get("enable_inference", app_settings.enable_inference);
    model.parameters.get("average_window_duration_ms", app_settings.average_window_duration_ms);
    model.parameters.get("minimum_count", app_settings.minimum_count);
    model.parameters.get("detection_threshold", app_settings.detection_threshold);
    model.parameters.get("suppression_count", app_settings.suppression_count);
    model.parameters.get("latency_ms", app_settings.latency_ms);
    model.parameters.get("activity_sensitivity", app_settings.activity_sensitivity);

    printf("Verbose inference output: %d\n", app_settings.verbose_inference_output);
    printf("Inference enabled: %d\n", app_settings.enable_inference);
    printf("Averaging window duration: %dms\n", app_settings.average_window_duration_ms);
    printf("Minimum averaging count: %d\n", app_settings.minimum_count);
    printf("Detection threshold: %d\n", app_settings.detection_threshold);
    printf("Supression count: %d samples\n", app_settings.suppression_count);
    printf("Minimum loop latency: %dms\n", app_settings.latency_ms);
    printf("Activity sensitivity: %f\n", app_settings.activity_sensitivity);


    return true;
}

/***************************************************************************//**
 * Initialize the ArduCAM
 ******************************************************************************/
static bool initialize_camera()
{
    sl_status_t status;

    // Initialize the camera
    const auto input_shape = model.input()->shape();

    arducam_config_t cam_config = ARDUCAM_DEFAULT_CONFIG;
    cam_config.image_resolution.width = input_shape[2];
    cam_config.image_resolution.height = input_shape[1];
    cam_config.data_format = input_shape[3] == 1 ?
        ARDUCAM_DATA_FORMAT_GRAYSCALE : ARDUCAM_DATA_FORMAT_RGB888;

    // Calculate the size required to buffer an image
    // NOTE: The buffer size may be different than the image size
    const uint32_t length_per_image = arducam_calculate_image_buffer_length(
        cam_config.data_format,
        cam_config.image_resolution.width,
        cam_config.image_resolution.height
    );

    // Allocate a "ping-pong" buffer (i.e. 2) for the image
    const uint32_t image_buffer_count = 2;
    const uint32_t image_buffer_length = length_per_image*image_buffer_count;
    uint8_t* image_buffer = (uint8_t*)malloc(image_buffer_length);
    if(image_buffer == nullptr)
    {
        printf("Failed to allocate camera buffer, size: %d\n", image_buffer_length);
        return false;
    }

    // Initialize the camera
    status = arducam_init(&cam_config, image_buffer, image_buffer_length);
    if(status != SL_STATUS_OK)
    {
        printf("Failed to initialize the camera, err: %u\n", status);
        return false;
    }

   // Start the image capturing DMA background
    status = arducam_start_capture();
    if(status != SL_STATUS_OK)
    {
        printf("Failed to start camera capture, err: %u\n", status);
        return false;
    }

    return true;
}

/***************************************************************************//**
 * Retrieve the next image from the camera
 ******************************************************************************/
static void retrieve_next_camera_image(uint8_t** image_data, uint32_t *image_size)
{
    for(;;)
    {
        sl_status_t status = arducam_get_next_image(image_data, image_size);

        if(status == SL_STATUS_IN_PROGRESS)
        {
            // NOTE: Unfortunately, the camera doesn't not have a way to interrupt the MCU
            //       once an image is ready. So we must periodically poll the camera to check
            //       its status. This can be done by continuously calling arducam_get_next_image()
            //       OR arducam_poll() from a timer interrupt handler.
            sl_sleeptimer_delay_millisecond(5);
            continue;
        }
        else if(status != SL_STATUS_OK)
        {
            printf("ERROR: Failed to retrieve image, err: %u\n", status);
            while (1)
            ;
        }

        dump_image(*image_data, *image_size);

        break;
    }
}

/***************************************************************************//**
 * Standardize the image and copy to model input tensor
 ******************************************************************************/
static void standardize_image_data(uint8_t* image_data, uint32_t image_size)
{
    const auto input_tensor = model.input(0);
    const auto& input_data = input_tensor->data;

    if(app_settings.samplewise_norm_rescale != 0)
    {
        // input_tensor = image_data * scaler
        mltk::scale_tensor(app_settings.samplewise_norm_rescale, image_data, input_data.f, image_size);
    }
    else if(app_settings.samplewise_norm_mean_and_std)
    {
        // input_tensor = (image_data - mean(image_data)) / std(image_data)
        mltk::samplewise_mean_std_tensor(image_data, input_data.f, image_size);
    }
    else if(input_tensor->type == kTfLiteInt8)
    {
        // Convert the uint8 image data to int8
        const uint8_t* src = image_data;
        int8_t* dst = input_data.int8;
        for(int remaining = image_size; remaining > 0; --remaining)
        {
            *dst++ = (int8_t)(*src++ - 128);
        }
    }
    else if(input_tensor->type == kTfLiteFloat32)
    {
        // Convert the uint8 image data to float32
        const uint8_t* src = image_data;
        float* dst = input_data.f;
        for(int remaining = image_size; remaining > 0; --remaining)
        {
            *dst++ = (float)(*src++);
        }
    }
    else
    {
            printf("ERROR: Invalid data type\n");
            while (1)
            ;
    }
}


/***************************************************************************//**
 * Processes the output from the output tensor
 ******************************************************************************/
static void process_inference_output()
{
    uint8_t result = 0;
    uint8_t score = 0;
    bool is_new_command = false;
    uint32_t current_time_stamp;

    // Get current time stamp needed by CommandRecognizer
    current_time_stamp = sl_sleeptimer_tick_to_ms(sl_sleeptimer_get_tick_count());

    TfLiteStatus process_status = command_recognizer->ProcessLatestResults(
        model.output(), current_time_stamp, &result, &score, &is_new_command);

    if (process_status == kTfLiteOk)
    {
        handle_result(current_time_stamp, result, score, is_new_command);
    }

}


/***************************************************************************//**
 * Handle inference result
 *
 * This function is called whenever we have a succesful inference result.
 *
 * @param current_time timestamp of the inference result.
 * @param result classification result, this is number >= 0.
 * @param score the score of the result. This is number represents the confidence
 *   of the result classification.
 * @param is_new_command true if the result is a new command, false otherwise.
 ******************************************************************************/
static void handle_result(int32_t current_time, int result, uint8_t score, bool is_new_command)
{
  const char *label = get_category_label(result);

  if (is_new_command)
  {
    printf("Detected class=%d label=%s score=%d @%ldms\n", result, label, score, current_time);
    fflush(stdout);
    sl_led_turn_on(&DETECTION_LED);
    sl_led_turn_off(&ACTIVITY_LED);
    detected_timeout = current_time + 1100;
  }
  else if (detected_timeout != 0 && current_time >= detected_timeout)
  {
    detected_timeout = 0;
    previous_score = score;
    previous_result = result;
    previous_score_timestamp = current_time;
    sl_led_turn_off(&DETECTION_LED);
  }

  if (detected_timeout == 0)
  {
    if (previous_score == 0)
    {
      previous_result = result;
      previous_score = score;
      previous_score_timestamp = current_time;
      return;
    }

    // Calculate the rate of difference in score between the two last results
    const int32_t time_delta = current_time - previous_score_timestamp;
    const int8_t score_delta = (int8_t)(score - previous_score);
    const float diff = (time_delta > 0) ? std::fabs(score_delta) / time_delta : 0.0f;

    previous_score = score;
    previous_score_timestamp = current_time;

    if (diff >= app_settings.activity_sensitivity || (previous_result != result))
    {
      previous_result = result;
      activity_timestamp = current_time + 500;
    }
    else if(current_time >= activity_timestamp)
    {
      activity_timestamp = 0;
      sl_led_turn_off(&ACTIVITY_LED);
    }

    if (activity_timestamp != 0)
    {
      if (current_time - activity_toggle_timestamp >= 100)
      {
        activity_toggle_timestamp = current_time;
        sl_led_toggle(&ACTIVITY_LED);
      }
    }
  }
}

/***************************************************************************//**
 * Dump camera image to JLink stream
 ******************************************************************************/
static void dump_image(const uint8_t* image_data, uint32_t image_length)
{
    bool connected = false;

    // Check if the Python script has connected
    jlink_stream::is_connected("image", &connected);
    if(connected)
    {
      jlink_stream::write_all("image", image_data, image_length);
    }
}




/***************************************************************************//**
 * Get the label for a certain category/class
 *
 * @param index
 *   index of the category/class
 *
 * @return
 *   pointer to the label string. The label is "?" if no corresponding label
 *   was found.
 ******************************************************************************/
const char * get_category_label(int index)
{
  if ((index >= 0) && (index < category_labels.size())) {
    return category_labels[index];
  } else {
    return "?";
  }
}
