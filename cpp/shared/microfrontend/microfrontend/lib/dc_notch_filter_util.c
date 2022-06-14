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
#include "microfrontend/lib/dc_notch_filter_util.h"


void DcNotchFilterFillConfigWithDefaults(struct DcNotchFilterConfig* config)
{
    config->enable_dc_notch_filter = 0; // DC Notch filter disabled by default
    config->coefficient = 0.95f;
}

void DcNotchFilterConfig(const struct DcNotchFilterConfig* config, struct DcNotchFilterState* state)
{
    state->enable_dc_notch_filter = config->enable_dc_notch_filter;
    state->dc_notch_coef = (int16_t)(config->coefficient*(1<<15));
    state->prev_in = 0;
    state->prev_out = 0;
}

