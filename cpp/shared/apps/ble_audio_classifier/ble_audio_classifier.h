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
#ifndef AUDIO_CLASSIFIER_H
#define AUDIO_CLASSIFIER_H

#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif


typedef void (*BleAudioClassifierDetectionCallback)(uint8_t, uint8_t);


/***************************************************************************//**
 * Initialize application.
 ******************************************************************************/
void ble_audio_classifier_init();
void ble_audio_classifier_start();
void ble_audio_classifier_stop();
void ble_audio_classifier_set_detection_callback(BleAudioClassifierDetectionCallback callback);


#ifdef __cplusplus
}
#endif

#endif // AUDIO_CLASSIFIER_H
