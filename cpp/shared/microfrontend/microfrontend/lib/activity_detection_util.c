/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file has been modified by Silicon Labs.
==============================================================================*/
#include "microfrontend/lib/activity_detection_util.h"

#include <math.h>

void ActivityDetectionFillConfigWithDefaults(struct ActivityDetectionConfig* config)
{
    config->enable_activation_detection = 0; // activity detection disabled by default
    config->alpha_a = 0.5; // fast filter
    config->alpha_b = 0.8; // slow filter
    config->arm_threshold = 0.75;
    config->trip_threshold = 0.8;
}

void ActivityDetectionConfig(const struct ActivityDetectionConfig* config, int num_channels, struct ActivityDetectionState* state)
{
    state->enable_activity_detection = config->enable_activation_detection;
    state->num_channels = num_channels;
    state->scale = (int)ceil(log2(num_channels));
    state->alpha_a = (int16_t)((config->alpha_a)*(1<<8)); // fast filter
    state->alpha_b =(int16_t)((config->alpha_b)*(1<<8)); // slow filter
    state->arm_threshold = (int16_t)((config->arm_threshold)*(1<<12));
    state->trip_threshold = (int16_t)((config->trip_threshold)*(1<<12));
    state->filter_a = 0;
    state->filter_b = 0;
    state->arm = 0;
    state->trip = 0;
}
