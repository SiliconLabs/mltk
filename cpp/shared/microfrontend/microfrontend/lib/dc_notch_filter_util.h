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
#ifndef DC_NOTCH_FILTER_UTIL_H_
#define DC_NOTCH_FILTER_UTIL_H_

#include "microfrontend/lib/utils.h"
#include "microfrontend/lib/dc_notch_filter.h"

#ifdef __cplusplus
extern "C" {
#endif

struct DcNotchFilterConfig {
    int enable_dc_notch_filter; // set to false (1) to enable DC notch filter
    float coefficient; // DC notch filter coefficient k in Q(16,15) format, H(z) = (1 - z^-1)/(1 - k*z^-1)
};

DLL_EXPORT void DcNotchFilterFillConfigWithDefaults(struct DcNotchFilterConfig* config);

DLL_EXPORT void DcNotchFilterConfig(const struct DcNotchFilterConfig* config, struct DcNotchFilterState* state);


#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // DC_NOTCH_FILTER_UTIL_H_
