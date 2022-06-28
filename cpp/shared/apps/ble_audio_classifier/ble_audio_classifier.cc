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
#include <cstdio>

#include "FreeRTOS.h"
#include "FreeRTOSConfig.h"
#include "task.h"
#include "event_groups.h"

#include "sl_power_manager.h"
#include "sl_status.h"
#include "sl_led.h"
#include "sl_simple_led_instances.h"
#include "ble_audio_classifier.h"

#include "ble_audio_classifier_config.h"
#include "sl_ml_audio_feature_generation.h"
#include "sl_ml_audio_feature_generation_config.h"
#include "sl_sleeptimer.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tflite_micro_model/tflite_micro_model.hpp"
#include "mltk_tflite_micro_helper.hpp"
#include "app.h"
#include "recognize_commands.h"


static void audio_classifier_task(void *arg);




static StackType_t task_stack[TASK_STACK_SIZE];
static StaticTask_t task_handle;
static EventGroupHandle_t event_handle;
static StaticEventGroup_t event_group;



static tflite::MicroMutableOpResolver<10> opcode_resolver;
static RecognizeCommands *command_recognizer = nullptr;
static mltk::TfliteMicroModel model;

static int32_t detected_timeout = 0;
static int32_t activity_timestamp = 0;
static int32_t activity_toggle_timestamp = 0;
static uint8_t previous_score = 0;
static int32_t previous_score_timestamp = 0;
static int previous_result = 0;

int category_count = 0;
static mltk::StringList category_labels;
static int category_label_count;
static BleAudioClassifierDetectionCallback detection_callback = nullptr;

// This is defined by the build scripts
// which converts the specified .tflite to a C array
extern "C" const uint8_t sl_tflite_model_array[];



static void handle_results(int32_t current_time, int result, uint8_t score, bool is_new_command);
static sl_status_t run_inference();
static sl_status_t process_output(const bool did_run_inference);



/***************************************************************************//**
 * Initialize audio classifier application.
 ******************************************************************************/
extern "C" void ble_audio_classifier_init(void)
{
  sl_status_t status;
  const uint8_t* model_flatbuffer = sl_tflite_model_array;

  printf("BLE Audio Classifier\n");


  // Register the accelerator if the TFLM lib was built with one
  mltk::mltk_tflite_micro_register_accelerator();

  // Register the basic kernels 
  opcode_resolver.AddConv2D();
  opcode_resolver.AddDepthwiseConv2D();
  opcode_resolver.AddMaxPool2D();
  opcode_resolver.AddReshape();
  opcode_resolver.AddFullyConnected();
  opcode_resolver.AddPad();
  opcode_resolver.AddAdd();
  opcode_resolver.AddMean();
  opcode_resolver.AddSoftmax();


  // Attempt to load the model using the arena size specified in the .tflite
  if(!model.load(model_flatbuffer, opcode_resolver))
  {
    printf("ERROR: Failed to load .tflite model\n");
    while(1)
      ;
  }


  model.print_summary();

  // Initialize the audio feature generation using the parameters
  // from the given .tflite model file
  if(!mltk_sl_ml_audio_feature_generation_load_parameters(model_flatbuffer))
  {
    printf("ERROR: Failed to load audio feature generator parameters from .tflite model\n");
    while(1)
      ;
  }
 
  // Load the other application-specific model parameters from the .tflite
  if(!ble_audio_classifier_config_load(model_flatbuffer))
  {
    printf("ERROR: Failed to load app parameters from .tflite model\n");
    while(1)
      ;
  }

  category_labels = CATEGORY_LABELS;
  category_label_count = CATEGORY_LABELS.size();

  status = sl_ml_audio_feature_generation_init();
  if(status != SL_STATUS_OK)
  {
    printf("ERROR: Failed to init audio feature generator\n");
    while(1)
      ;
  }

  // Instantiate CommandRecognizer  
  static RecognizeCommands static_recognizer(model.error_reporter(), SMOOTHING_WINDOW_DURATION_MS,
      DETECTION_THRESHOLD, SUPPRESION_TIME_MS, MINIMUM_DETECTION_COUNT, IGNORE_UNDERSCORE_LABELS);
  command_recognizer = &static_recognizer;

  const TfLiteTensor* input = model.input();
  const TfLiteTensor* output = model.output();

  // Validate model tensors
  if ((output->dims->size == 2) && (output->dims->data[0] == 1)) {
    category_count = output->dims->data[1];
  } else {
    printf("ERROR: Invalid output tensor shape\n"
           "expecting an output tensor of shape [1,x]\n"
           "where x is the number of classification results\n");
    while (1)
      ;
  }

  if (category_count != category_label_count) {
    printf("WARNING: Number of categories(%d) is not equal to the number of labels(%d).\n"
           "Make sure that CATEGORY_LABELS is configured correctly for the model in use.\n",
           category_count, category_label_count);
  }

  if (!(input->type == kTfLiteInt8 || input->type == kTfLiteUInt16 || input->type == kTfLiteFloat32)) {
    printf("ERROR: Invalid input tensor type.\n"
           "Application requires input and output tensors to be of type int8, uint16, or float32.\n");
    while (1)
      ;
  }

  if (!(output->type == kTfLiteInt8 || output->type != kTfLiteFloat32)) {
    printf("ERROR: Invalid output tensor type.\n"
           "Application requires input and output tensors to be of type int8 or float32.\n");
    while (1)
      ;
  }

  // Add EM1 requirement to allow microphone sampling 
  sl_power_manager_add_em_requirement(SL_POWER_MANAGER_EM1);


  event_handle = xEventGroupCreateStatic( &event_group );
  ble_audio_classifier_stop();

  printf("NOTE: Audio classification will start when a bluetooth client connects\n");

  xTaskCreateStatic(
    audio_classifier_task,
    "audio_classifier",
    TASK_STACK_SIZE,
    NULL,
    TASK_PRIORITY,
    task_stack,
    &task_handle
  );
}


/***************************************************************************//**
 * Start the audio classification loop
 ******************************************************************************/
extern "C" void ble_audio_classifier_start()
{
    printf("Starting audio classifier\n");
    xEventGroupSetBits(event_handle, 0x01);
}


/***************************************************************************//**
 * Stop the audio classification loop
 ******************************************************************************/
extern "C" void ble_audio_classifier_stop()
{
    printf("Stopping audio classifier\n");
    xEventGroupClearBits(event_handle, 0x01);
    sl_led_turn_off(&DETECTION_LED);
    sl_led_turn_off(&ACTIVITY_LED);
}


/***************************************************************************//**
 * Set the detection callback
 ******************************************************************************/
extern "C" void ble_audio_classifier_set_detection_callback(BleAudioClassifierDetectionCallback callback)
{
  detection_callback = callback;
}


#ifdef SL_CATALOG_KERNEL_PRESENT
/***************************************************************************//**
 * Audio classifier task function
 *
 * This function is executed by a Micrium OS task and does not return.
 *
 * @param arg ignored
 ******************************************************************************/
void audio_classifier_task(void *arg)
{
  (void)&arg;
  while(1) 
  {
    // Wait for audio classifier to start (if it hasn't already)
    xEventGroupWaitBits(event_handle, 0x01, pdFALSE, pdTRUE, portMAX_DELAY);

    app_process_action();
       
    // Delay task in order to do periodic inference
    vTaskDelay(pdMS_TO_TICKS(INFERENCE_INTERVAL_MS));
  }
}
#endif

/***************************************************************************//**
 * Run a single application loop
 *
 * This function is executed by either the RTOS task or main.c loop
 ******************************************************************************/
extern "C" void app_process_action()
{
  static uint32_t prev_loop_timestamp = 0;

  uint32_t current_timestamp = sl_sleeptimer_tick_to_ms(sl_sleeptimer_get_tick_count());

  if((current_timestamp - prev_loop_timestamp) >= INFERENCE_INTERVAL_MS)
  {
    // Store the current timestamp before we run the audio feature generator
    // and do model inference
    command_recognizer->base_timestamp_ = current_timestamp;
    // Perform a word detection
    prev_loop_timestamp = current_timestamp;

    // Process the audio buffer
    sl_ml_audio_feature_generation_update_features();

  
    // Determine if we should run inference
    // If the activity detection block is disabled, then always run inference
    // If the activity detection block is enabled, then ensure there is activity before running inference
    const bool should_run_inference = 
      (!SL_ML_FRONTEND_ACTIVITY_DETECTION_ENABLE || 
      (sl_ml_audio_feature_generation_activity_detected() == SL_STATUS_OK));

    if(should_run_inference)
    {
      // Execute the processed audio in the ML model
      run_inference();
    }
   
    // Process the ML model results
    // NOTE: We do this even if we didn't run inference.
    //       This way, the LEDs blink correctly
    process_output(should_run_inference);
  }
}


/***************************************************************************//**
 * Run model inference 
 * 
 * Copies the currently available data from the feature_buffer into the input 
 * tensor and runs inference, updating the global output tensor.
 *
 * @return
 *   SL_STATUS_OK on success, other value on failure.
 ******************************************************************************/
static sl_status_t run_inference()
{
  // Update model input tensor
  sl_status_t status = sl_ml_audio_feature_generation_fill_tensor(model.input());
  if (status != SL_STATUS_OK){
    return SL_STATUS_FAIL;
  }
  // Run the model on the spectrogram input and make sure it succeeds.
  if (!model.invoke()) {
    return SL_STATUS_FAIL;
  }

  return SL_STATUS_OK;
}

/***************************************************************************//**
 * Processes the output from the output tensor
 *
 * @return
 *   SL_STATUS_OK on success, other value on failure.
 ******************************************************************************/
static sl_status_t process_output(const bool did_run_inference){
  // Determine whether a command was recognized based on the output of inference
  uint8_t result = 0;
  uint8_t score = 0;
  bool is_new_command = false;
  sl_status_t status = SL_STATUS_OK;
  TfLiteStatus process_status = kTfLiteOk;
  const uint32_t current_timestamp = sl_sleeptimer_tick_to_ms(sl_sleeptimer_get_tick_count());

  if(did_run_inference)
      process_status = command_recognizer->ProcessLatestResults(
        model.output(), 
        current_timestamp, 
        &result, 
        &score, 
        &is_new_command
      );

  if (process_status == kTfLiteOk) {
    handle_results(current_timestamp, result, score, is_new_command);
  } else {
    status = SL_STATUS_FAIL;
  }

  return status;
}

/***************************************************************************//**
 * Handle inference result
 *
 * This function is called whenever we have a succesfull inference result.
 *
 * @param current_time timestamp of the inference result.
 * @param result classification result, this is number >= 0.
 * @param score the score of the result. This is number represents the confidence
 *   of the result classification.
 * @param is_new_command true if the result is a new command, false otherwise.
 ******************************************************************************/
static void handle_results(int32_t current_time, int result, uint8_t score, bool is_new_command) {
  const char *label = SL_TFLITE_MODEL_CLASSES[result];

  if (is_new_command) {
    // Reset the AFG internal state so we can detect a new keyword
    // NOTE: Alternatively, the "suppression" setting can be increased to add a delay
    //       until processing states again (this effectively clears the audio buffer)
    sl_ml_audio_feature_generation_reset(); 
    
    printf("Detected class=%d label=%s score=%d @%ldms\n", result, label, score, current_time);
    if(detection_callback != nullptr)
    {
      detection_callback((uint8_t)result, score);
    }

    fflush(stdout);
    sl_led_turn_on(&DETECTION_LED);
    sl_led_turn_off(&ACTIVITY_LED);
    activity_timestamp = 0;
    detected_timeout = current_time + 1200;
  } else if (detected_timeout != 0 && current_time >= detected_timeout) {
    detected_timeout = 0;
    previous_score = score;
    previous_result = result;
    previous_score_timestamp = current_time;
    sl_led_turn_off(&DETECTION_LED);
  }

  // If we're using the activity detection block,
  // then inference is only done when activity is detected
  // in the audio stream. In this case, we control
  // the LEDs based on static timeouts
  if(SL_ML_FRONTEND_ACTIVITY_DETECTION_ENABLE)
  {
    if (detected_timeout == 0 && score > 0 && previous_result != result) {
      activity_timestamp = current_time + 1000;
    } else if(current_time >= activity_timestamp) {
      activity_timestamp = 0;
      sl_led_turn_off(&ACTIVITY_LED);
    }

    if (activity_timestamp != 0) {
      if (current_time - activity_toggle_timestamp >= 100) {
        activity_toggle_timestamp = current_time;
        sl_led_toggle(&ACTIVITY_LED);
      }
    }
    return;
  }

  // If the activity detection block is NOT used,
  // then inference is also done a a specific interval.
  // In this case, we control the LEDs based on the scores 
  // returned by the inference
  if (detected_timeout == 0) {
    if (previous_score == 0) {
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

    if (diff >= SENSITIVITY || (previous_result != result && score > 255/2)) {
      previous_result = result;
      activity_timestamp = current_time + 500;
    } else if(current_time >= activity_timestamp) {
      activity_timestamp = 0;
      sl_led_turn_off(&ACTIVITY_LED);
    }

    if (activity_timestamp != 0) {
      if (current_time - activity_toggle_timestamp >= 100) {
        activity_toggle_timestamp = current_time;
        sl_led_toggle(&ACTIVITY_LED);
      }
    }
  }
}

