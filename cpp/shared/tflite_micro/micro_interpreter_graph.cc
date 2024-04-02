/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/micro_interpreter_graph.h"

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/flatbuffer_utils.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.h"

#include "mltk_tflite_micro_helper.hpp"

namespace tflite {
namespace {

const char* OpNameFromRegistration(const TFLMRegistration* registration) {
  if (registration->builtin_code == BuiltinOperator_CUSTOM) {
    return registration->custom_name;
  } else {
    return EnumNameBuiltinOperator(BuiltinOperator(registration->builtin_code));
  }
}

}  // namespace

MicroInterpreterGraph::MicroInterpreterGraph(
    TfLiteContext* context, const Model* model, MicroAllocator* allocator,
    MicroResourceVariables* resource_variables)
    : context_(context),
      model_(model),
      allocator_(allocator),
      current_subgraph_index_(0),
      resource_variables_(resource_variables) {
  if (model != nullptr) {
    subgraphs_ = model->subgraphs();
  }
}

MicroInterpreterGraph::~MicroInterpreterGraph() {
  MLTK_FREE_PROFILERS();
  MLTK_RECORD_RESET();
}

TfLiteStatus MicroInterpreterGraph::InitSubgraphs() {
  auto context = context_;
  int previous_subgraph_idx = current_subgraph_index_;

  MLTK_RECORD_BEGIN_SECTION();

  for (size_t subgraph_idx = 0; subgraph_idx < subgraphs_->size();
       subgraph_idx++) {
    current_subgraph_index_ = subgraph_idx;
    uint32_t operators_size = NumSubgraphOperators(model_, subgraph_idx);
    for (size_t i = 0; i < operators_size; ++i) {
      TfLiteNode* node =
          &(subgraph_allocations_[subgraph_idx].node_and_registrations[i].node);
      const TFLMRegistration* registration = subgraph_allocations_[subgraph_idx]
                                                 .node_and_registrations[i]
                                                 .registration;
      size_t init_data_size;
      const char* init_data;
      if (registration->builtin_code == BuiltinOperator_CUSTOM) {
        init_data = reinterpret_cast<const char*>(node->custom_initial_data);
        init_data_size = node->custom_initial_data_size;
      } else {
        init_data = reinterpret_cast<const char*>(node->builtin_data);
        init_data_size = 0;
      }
      if (registration->init) {
        MLTK_SET_CURRENT_KERNEL(i, registration->builtin_code);
        MLTK_RECORD_BEGIN_LAYER(i, node);
        node->user_data =
            registration->init(context_, init_data, init_data_size);
        MLTK_RECORD_END_LAYER();
        MLTK_CLEAR_CURRENT_KERNEL();
      }
    }
  }
  current_subgraph_index_ = previous_subgraph_idx;
  MLTK_RECORD_END_SECTION();

  return kTfLiteOk;
}

TfLiteStatus MicroInterpreterGraph::PrepareSubgraphs() {
  TfLiteStatus retval = kTfLiteOk;
  auto context = context_;
  auto& logger = mltk::get_logger();
  int previous_subgraph_idx = current_subgraph_index_;
  #ifdef TFLITE_MICRO_RECORDER_ENABLED
  const auto buffer_allocator = reinterpret_cast<SingleArenaBufferAllocator*>(allocator_->non_persistent_buffer_allocator_);
  #endif // TFLITE_MICRO_RECORDER_ENABLED

  MLTK_RECORD_BEGIN_SECTION();

  for (size_t subgraph_idx = 0; subgraph_idx < subgraphs_->size();
       subgraph_idx++) {
    current_subgraph_index_ = subgraph_idx;
    uint32_t operators_size = NumSubgraphOperators(model_, subgraph_idx);
    MLTK_ALLOCATE_PROFILERS(subgraph_idx, operators_size)
    for (size_t i = 0; i < operators_size; ++i) {
      TfLiteNode* node =
          &(subgraph_allocations_[subgraph_idx].node_and_registrations[i].node);
      const TFLMRegistration* registration =
          subgraph_allocations_[subgraph_idx]
              .node_and_registrations[i]
              .registration;
      MLTK_SET_CURRENT_KERNEL(i, registration->builtin_code);
    
      if (registration->prepare != nullptr) {
        MLTK_RECORD_BEGIN_LAYER(i, node);
        #ifdef TFLITE_MICRO_RECORDER_ENABLED
        const uint8_t* previous_tail = buffer_allocator->tail_;
        #endif // TFLITE_MICRO_RECORDER_ENABLED

        TfLiteStatus prepare_status = registration->prepare(context_, node);

        #ifdef TFLITE_MICRO_RECORDER_ENABLED
        const int used_temp = (buffer_allocator->temp_ - buffer_allocator->head_);
        const int used_persistent = (previous_tail - buffer_allocator->tail_);
        MLTK_RECORD_LAYER_PARAM("temp_memory_used", used_temp);
        MLTK_RECORD_LAYER_PARAM("persistent_memory_used", used_persistent);
        #endif // TFLITE_MICRO_RECORDER_ENABLED

        MLTK_RECORD_END_LAYER();


        if(!mltk::TfliteMicroKernelMessages::have_messages() && 
          prepare_status != kTfLiteOk
        )
        {
          MLTK_KERNEL_UNSUPPORTED_MSG("Failed to prepare with status %d", prepare_status);
        }

        mltk::TfliteMicroKernelMessages::flush(logging::Error);

        if (prepare_status != kTfLiteOk) {

          retval = kTfLiteError;
          continue;
        }
      }

      allocator_->FinishPrepareNodeAllocations(/*node_id=*/i);
      MLTK_CLEAR_CURRENT_KERNEL();
      MLTK_REGISTER_PROFILER(
        subgraph_idx, 
        i, 
        registration->builtin_code, 
        subgraph_allocations_[subgraph_idx].node_and_registrations[i]
      );
    }
  }
  current_subgraph_index_ = previous_subgraph_idx;
  MLTK_RECORD_END_SECTION();

  return retval;
}

TfLiteStatus MicroInterpreterGraph::ResetSubgraphs() {
  int previous_subgraph_idx = current_subgraph_index_;

  for (size_t subgraph_idx = 0; subgraph_idx < subgraphs_->size();
       subgraph_idx++) {
    current_subgraph_index_ = subgraph_idx;
    uint32_t operators_size = NumSubgraphOperators(model_, subgraph_idx);
    for (size_t i = 0; i < operators_size; ++i) {
      TfLiteNode* node =
          &(subgraph_allocations_[subgraph_idx].node_and_registrations[i].node);
      const TFLMRegistration* registration = subgraph_allocations_[subgraph_idx]
                                                 .node_and_registrations[i]
                                                 .registration;
      // registration is allocated outside the interpreter, so double check to
      // make sure it's not nullptr;
      if (registration != nullptr && registration->reset != nullptr) {
        registration->reset(context_, node->user_data);
      }
    }
  }
  current_subgraph_index_ = previous_subgraph_idx;

  return kTfLiteOk;
}

TfLiteStatus MicroInterpreterGraph::FreeSubgraphs() {
  int previous_subgraph_idx = current_subgraph_index_;

  for (size_t subgraph_idx = 0; subgraph_idx < subgraphs_->size();
       subgraph_idx++) {
    current_subgraph_index_ = subgraph_idx;
    uint32_t operators_size = NumSubgraphOperators(model_, subgraph_idx);
    for (size_t i = 0; i < operators_size; ++i) {
      TfLiteNode* node =
          &(subgraph_allocations_[subgraph_idx].node_and_registrations[i].node);
      const TFLMRegistration* registration = subgraph_allocations_[subgraph_idx]
                                                 .node_and_registrations[i]
                                                 .registration;
      // registration is allocated outside the interpreter, so double check to
      // make sure it's not nullptr;
      if (registration != nullptr && registration->free != nullptr) {
        registration->free(context_, node->user_data);
      }
    }
  }
  current_subgraph_index_ = previous_subgraph_idx;

  return kTfLiteOk;
}

TfLiteStatus MicroInterpreterGraph::InvokeSubgraph(int subgraph_idx) {
  auto& logger = mltk::get_logger();
  auto context = context_;
  int previous_subgraph_idx = current_subgraph_index_;
  current_subgraph_index_ = subgraph_idx;

  if (static_cast<size_t>(subgraph_idx) >= subgraphs_->size()) {
    MicroPrintf("Accessing subgraph %d but only %d subgraphs found",
                subgraph_idx, subgraphs_->size());
    return kTfLiteError;
  }
  MLTK_RECORD_BEGIN_SECTION();
  MLTK_INVOKE_PROCESSING_CALLBACK();
  MLTK_START_INFERENCE_PROFILER(subgraph_idx);
  uint32_t operators_size = NumSubgraphOperators(model_, subgraph_idx);
  for (size_t i = 0; i < operators_size; ++i) {
    auto& node_and_registration = subgraph_allocations_[subgraph_idx].node_and_registrations[i];
    TfLiteNode* node = &node_and_registration.node;
    const TFLMRegistration* registration = node_and_registration.registration;

// This ifdef is needed (even though ScopedMicroProfiler itself is a no-op with
// -DTF_LITE_STRIP_ERROR_STRINGS) because the function OpNameFromRegistration is
// only defined for builds with the error strings.
#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
    ScopedMicroProfiler scoped_profiler(
        OpNameFromRegistration(registration),
        reinterpret_cast<MicroProfilerInterface*>(context_->profiler));
#endif

    TFLITE_DCHECK(registration->invoke);
    MLTK_SET_CURRENT_KERNEL(i, registration->builtin_code);
    MLTK_RECORD_BEGIN_LAYER(i, node);
    MLTK_START_OP_PROFILER(subgraph_idx, i);
    TfLiteStatus invoke_status = registration->invoke(context_, node);
    MLTK_STOP_OP_PROFILER(subgraph_idx, i)
    MLTK_INVOKE_LAYER_CALLBACK(node_and_registration, invoke_status);
    MLTK_RECORD_END_LAYER();
    MLTK_CLEAR_CURRENT_KERNEL();

    // All TfLiteTensor structs used in the kernel are allocated from temp
    // memory in the allocator. This creates a chain of allocations in the
    // temp section. The call below resets the chain of allocations to
    // prepare for the next call.
    allocator_->ResetTempAllocations();

    if (invoke_status == kTfLiteError) {
      logger.error("Op%d-%s not supported: Failed to invoke with status %d",
                  i, OpNameFromRegistration(registration), invoke_status);
      return kTfLiteError;
    } else if (invoke_status != kTfLiteOk) {
      return invoke_status;
    }

    MLTK_INVOKE_PROCESSING_CALLBACK();
  }
  MLTK_STOP_INFERENCE_PROFILER(subgraph_idx);
  MLTK_RECORD_END_SECTION();
  current_subgraph_index_ = previous_subgraph_idx;
  return kTfLiteOk;
}

TfLiteStatus MicroInterpreterGraph::ResetVariableTensors() {
  for (size_t subgraph_idx = 0; subgraph_idx < subgraphs_->size();
       subgraph_idx++) {
    const SubGraph* subgraph = (*subgraphs_)[subgraph_idx];
    for (size_t i = 0; i < subgraph->tensors()->size(); ++i) {
      auto* tensor = subgraph->tensors()->Get(i);
      if (tensor->is_variable()) {
        size_t buffer_size;
        TF_LITE_ENSURE_STATUS(TfLiteEvalTensorByteLength(
            &subgraph_allocations_[subgraph_idx].tensors[i], &buffer_size));

        int value = 0;
        if (tensor->type() == tflite::TensorType_INT8) {
          value = tensor->quantization()->zero_point()->Get(0);
        }
        memset(subgraph_allocations_[subgraph_idx].tensors[i].data.raw, value,
               buffer_size);
      }
    }
  }
  if (resource_variables_ != nullptr) {
    resource_variables_->ResetAll();
  }

  return kTfLiteOk;
}

int MicroInterpreterGraph::NumSubgraphs() {
  return model_->subgraphs()->size();
}

void MicroInterpreterGraph::SetSubgraphAllocations(
    SubgraphAllocations* subgraph_allocations) {
  subgraph_allocations_ = subgraph_allocations;
}

size_t MicroInterpreterGraph::NumSubgraphInputs(int subgraph_idx) {
  return model_->subgraphs()->Get(subgraph_idx)->inputs()->size();
}

TfLiteEvalTensor* MicroInterpreterGraph::GetSubgraphInput(int subgraph_idx,
                                                          int input_idx) {
  int tensor_idx =
      model_->subgraphs()->Get(subgraph_idx)->inputs()->Get(input_idx);
  return &subgraph_allocations_[subgraph_idx].tensors[tensor_idx];
}

size_t MicroInterpreterGraph::NumSubgraphOutputs(int subgraph_idx) {
  return model_->subgraphs()->Get(subgraph_idx)->outputs()->size();
}

TfLiteEvalTensor* MicroInterpreterGraph::GetSubgraphOutput(int subgraph_idx,
                                                           int output_idx) {
  int tensor_idx =
      model_->subgraphs()->Get(subgraph_idx)->outputs()->Get(output_idx);
  return &subgraph_allocations_[subgraph_idx].tensors[tensor_idx];
}

}  // namespace tflite
