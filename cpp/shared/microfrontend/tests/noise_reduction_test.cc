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
==============================================================================*/
#include <stdio.h>

#include "microfrontend/lib/noise_reduction.h"

#include "microfrontend/lib/noise_reduction_util.h"
#include "gtest/gtest.h"

namespace {

const int kNumChannels = 2;

// Test noise reduction using default config values.
class NoiseReductionTestConfig {
 public:
  NoiseReductionTestConfig() {
    config_.enable_noise_reduction = 1;
    config_.smoothing_bits = 10;
    config_.even_smoothing = 0.025;
    config_.odd_smoothing = 0.06;
    config_.min_signal_remaining = 0.05;
  }

  struct NoiseReductionConfig config_;
};

}  // namespace


TEST(NoiseReduction, estNoiseReductionEstimate) {
  NoiseReductionTestConfig config;
  struct NoiseReductionState state;
  EXPECT_TRUE(
      NoiseReductionPopulateState(&config.config_, &state, kNumChannels));

  uint32_t signal[] = {247311, 508620};
  NoiseReductionApply(&state, signal);

  const uint32_t expected[] = {6321887, 31248341};
  EXPECT_EQ(state.num_channels,
                          sizeof(expected) / sizeof(expected[0]));
  int i;
  for (i = 0; i < state.num_channels; ++i) {
    EXPECT_EQ(state.estimate[i], expected[i]);
  }

  NoiseReductionFreeStateContents(&state);
}

TEST(NoiseReduction, TestNoiseReduction) {
  NoiseReductionTestConfig config;
  struct NoiseReductionState state;
  EXPECT_TRUE(
      NoiseReductionPopulateState(&config.config_, &state, kNumChannels));

  uint32_t signal[] = {247311, 508620};
  NoiseReductionApply(&state, signal);

  const uint32_t expected[] = {241137, 478104};
  EXPECT_EQ(state.num_channels,
                          sizeof(expected) / sizeof(expected[0]));
  int i;
  for (i = 0; i < state.num_channels; ++i) {
    EXPECT_EQ(signal[i], expected[i]);
  }

  NoiseReductionFreeStateContents(&state);
}

