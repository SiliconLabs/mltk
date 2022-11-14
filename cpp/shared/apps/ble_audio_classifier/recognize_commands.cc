/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <cstdio>
#include <limits>
#include "recognize_commands.h"
#include "sl_ml_audio_feature_generation_config.h"


RecognizeCommands::RecognizeCommands(int32_t average_window_duration_ms,
                                     uint8_t detection_threshold,
                                     int32_t suppression_ms,
                                     int32_t minimum_count,
                                     bool ignore_underscore)
  : base_timestamp_(0),
  average_window_duration_ms_(average_window_duration_ms),
  detection_threshold_(detection_threshold),
  suppression_ms_(suppression_ms),
  minimum_count_(minimum_count),
  ignore_underscore_(ignore_underscore)
{
  previous_top_label_index_ = 0;
  previous_top_label_time_ = 0;
}

TfLiteStatus RecognizeCommands::ProcessLatestResults(
  const TfLiteTensor* latest_results, const int32_t current_time_ms,
  uint8_t* found_command_index, uint8_t* score, bool* is_new_command)
{
  int8_t current_top_index = 0;
  int32_t current_top_score = 0;
  // Add the latest results to the head of the queue.
  uint8_t converted_scores[MAX_CATEGORY_COUNT];


  if ((latest_results->dims->size != 2)
      || (latest_results->dims->data[0] != 1)
      || (latest_results->dims->data[1] != SL_TFLITE_MODEL_CLASS_COUNT)) {
   MicroPrintf("The results for recognition should contain %d elements, but there are "
      "%d in an %d-dimensional shape",
      SL_TFLITE_MODEL_CLASS_COUNT, latest_results->dims->data[1],
      latest_results->dims->size);
    return kTfLiteError;
  }

  if ((!previous_results_.empty())
      && (current_time_ms < previous_results_.front().time_)) {
    MicroPrintf("Results must be fed in increasing time order, but received a "
      "timestamp of %d that was earlier than the previous one of %d",
      current_time_ms, previous_results_.front().time_);
    return kTfLiteError;
  }

  // Convert the model output from float32 to uint8
  if(latest_results->type == kTfLiteFloat32)
  {
    for(int i = 0; i < SL_TFLITE_MODEL_CLASS_COUNT; ++i)
    {
      converted_scores[i] = (uint8_t)(latest_results->data.f[i] *255);
    }
  }
  // Convert the model output from int8 to uint8
  else if(latest_results->type == kTfLiteInt8)
  {
    for(int i = 0; i < SL_TFLITE_MODEL_CLASS_COUNT; ++i)
    {
      converted_scores[i] = (uint8_t)(latest_results->data.int8[i] + 128);
    }
  }
  else
  {
      MicroPrintf("Unsupported output tensor data type, must be int8 or float32");
      return kTfLiteError;
  }

  if(VERBOSE_MODEL_OUTPUT_LOGS) {
    char buffer[256];
    char *ptr = buffer;

    int diff = current_time_ms - base_timestamp_;
    ptr += sprintf(ptr, "[%6ld] (%3d) ", current_time_ms, diff);
    for (int i = 0; i < SL_TFLITE_MODEL_CLASS_COUNT; ++i) {
      ptr += sprintf(ptr, "%3ld ", converted_scores[i]);
    }
    *ptr++ = 0;
    puts(buffer);
    fflush(stdout);
  }


  // If the min count is 0, then disable averaging and only consider the latest result
  if(minimum_count_ == 0) {
    // Find the current highest scoring category.
    for (int i = 0; i < SL_TFLITE_MODEL_CLASS_COUNT; ++i) {
      if (converted_scores[i] > current_top_score) {
        current_top_score = converted_scores[i];
        current_top_index = i;
      }
    }
  } else {
      // Add the latest results to the head of the queue.
    previous_results_.push_back({current_time_ms, converted_scores});

    // Prune any earlier results that are too old for the averaging window.
    const int64_t time_limit = current_time_ms - average_window_duration_ms_;
    while ((!previous_results_.empty())
          && previous_results_.front().time_ < time_limit) {
      previous_results_.pop_front();
    }

    // If there are too few results, assume the result will be unreliable and
    // bail.
    static int consecutive_min_count = 0;
    const int32_t how_many_results = previous_results_.size();
    if ((how_many_results < minimum_count_)) {
      ++consecutive_min_count;
      if(consecutive_min_count % 10 == 0)
      {
        printf("Too few samples for averaging. This likely means the inference loop is taking too long.\n");
        printf("Either decrease the 'minimum_count' and/or increase 'average_window_duration_ms'\n");
      }
      *found_command_index = previous_top_label_index_;
      *score = 0;
      *is_new_command = false;
      return kTfLiteOk;
    }
    consecutive_min_count = 0;

    // Calculate the average score across all the results in the window.
    uint32_t average_scores[SL_TFLITE_MODEL_CLASS_COUNT];
    for (int offset = 0; offset < previous_results_.size(); ++offset) {
      // Iterates the amount of times to achieve average_window_duration
      PreviousResultsQueue::Result previous_result =
        previous_results_.from_front(offset);
      const uint8_t* scores = previous_result.scores;
      for (int i = 0; i < SL_TFLITE_MODEL_CLASS_COUNT; ++i) {
        if (offset == 0) {
          average_scores[i] = scores[i];
        } else {
          average_scores[i] += scores[i];
        }
      }
    }

    for (int i = 0; i < SL_TFLITE_MODEL_CLASS_COUNT; ++i) {
      average_scores[i] /= how_many_results;
    }

    // Find the current highest scoring category.
    for (int i = 0; i < SL_TFLITE_MODEL_CLASS_COUNT; ++i) {
      if (average_scores[i] > current_top_score) {
        current_top_score = average_scores[i];
        current_top_index = i;
      }
    }
  }

  const char *current_top_label = SL_TFLITE_MODEL_CLASSES[current_top_index];

  // If we've recently had another label trigger, assume one that occurs too
  // soon afterwards is a bad result.
  int64_t time_since_last_top;
  time_since_last_top = current_time_ms - previous_top_label_time_;


  // If a per class, detection threshold list was given in the model parameters
  // then use that, otherwise default to the global detection threshold
  int detection_threshold = detection_threshold_;
  if(SL_TFLITE_DETECTION_THRESHOLD_LIST.size() > 0)
  {
    detection_threshold = SL_TFLITE_DETECTION_THRESHOLD_LIST[current_top_index];
  } 

  if ((current_top_score > detection_threshold) && 
     (ignore_underscore_ && current_top_label[0] != '_') &&
     ((current_top_index != previous_top_label_index_)
      || (time_since_last_top > suppression_ms_))) {
    previous_top_label_index_ = current_top_index;
    previous_top_label_time_ = current_time_ms;
    *is_new_command = true;
  } else {
    *is_new_command = false;
  }

  *found_command_index = current_top_index;
  *score = current_top_score;
  return kTfLiteOk;
}
