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
#include "tensorflow/lite/kernels/internal/reference/pooling.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/pooling.h"
#include "CMSIS/NN/Include/arm_nnfunctions.h"
#include "tensorflow/lite/micro/micro_log.h"


namespace tflite {

namespace {


struct CmisOpDataPooling
{
  OpDataPooling reference_op_data;
  int buffer_idx;
};


TfLiteStatus WrappedAveragePoolingPrepare(TfLiteContext* context, TfLiteNode* node) {
  TfLiteStatus status = PoolingPrepare(context, node);
  CmisOpDataPooling* data = static_cast<CmisOpDataPooling*>(node->user_data);
  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kPoolingOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);


  if(status == kTfLiteOk && output->type == kTfLiteInt8)
  {
    const int output_width = output->dims->data[2];
    const int channels = output->dims->data[3];

    // This is added so the tensor arena size is consistent with
    // what will be required on the embedded device
    int scratch_buffer_size = arm_avgpool_s8_get_buffer_size(
      output_width,
      channels);

    data->buffer_idx = -1;
    if(scratch_buffer_size > 0)
    {
      TF_LITE_ENSURE_STATUS(
        context->RequestScratchBufferInArena(
                  context, scratch_buffer_size, &data->buffer_idx));
    }


  }

  micro_context->DeallocateTempTfLiteTensor(output);

  return status;
}

TfLiteStatus AverageEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  const auto cmsis_data =
      static_cast<const CmisOpDataPooling*>(node->user_data);
  const auto data = &cmsis_data->reference_op_data;

  const TfLiteEvalTensor* input =
      micro::GetEvalInput(context, node, kPoolingInputTensor);
  TfLiteEvalTensor* output =
      micro::GetEvalOutput(context, node, kPoolingOutputTensor);

  // Inputs and outputs share the same type, guaranteed by the converter.
  switch (input->type) {
    case kTfLiteFloat32:
      AveragePoolingEvalFloat(context, node, params, data, input, output);
      break;
    case kTfLiteInt8:
      if(cmsis_data->buffer_idx >= 0) context->GetScratchBuffer(context, cmsis_data->buffer_idx);
      AveragePoolingEvalQuantized<int8_t>(context, node, params, data, input,
                                          output);
      break;
    case kTfLiteInt16:
      AveragePoolingEvalQuantized<int16_t>(context, node, params, data, input,
                                           output);
      break;
    default:
      MicroPrintf("Input type %s is not currently supported",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus MaxEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataPooling* data =
      static_cast<const OpDataPooling*>(node->user_data);

  const TfLiteEvalTensor* input =
      micro::GetEvalInput(context, node, kPoolingInputTensor);
  TfLiteEvalTensor* output =
      micro::GetEvalOutput(context, node, kPoolingOutputTensor);

  switch (input->type) {
    case kTfLiteFloat32:
      MaxPoolingEvalFloat(context, node, params, data, input, output);
      break;
    case kTfLiteInt8:
      MaxPoolingEvalQuantized<int8_t>(context, node, params, data, input,
                                      output);
      break;
    case kTfLiteInt16:
      MaxPoolingEvalQuantized<int16_t>(context, node, params, data, input,
                                       output);
      break;
    default:
      MicroPrintf("Type %s not currently supported.",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(CmisOpDataPooling));
}

}  // namespace

TFLMRegistration Register_AVERAGE_POOL_2D() {
  return tflite::micro::RegisterOp(Init, WrappedAveragePoolingPrepare, AverageEval);
}

TFLMRegistration Register_MAX_POOL_2D() {
  return tflite::micro::RegisterOp(Init, PoolingPrepare, MaxEval);
}

}  // namespace tflite
