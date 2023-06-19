
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "CMSIS/NN/Include/arm_nnfunctions.h"

#include "sl_mvp_config.h"
#include "sl_mvp_ml_conv2d.h"

namespace tflite {
namespace sl {
namespace conv2d {

constexpr int kInputTensor  = 0;
constexpr int kFilterTensor = 1;
constexpr int kBiasTensor   = 2;
constexpr int kOutputTensor = 0;

// Conv is quantized along dimension 0 of filter tensor.
// https://www.tensorflow.org/lite/performance/quantization_spec
constexpr int kConvQuantizedDimension = 0;

enum op_support { kMvp, kCmsisNN, kTFLMrefF32, kTFLMrefI8 };

struct OpData {
  op_support  supported;
  float       activation_min_f32;
  float       activation_max_f32;
  int         scratch_buffer_index;
  sli_mvp_ml_conv2d_s8_params_t op_params;

  // CMSIS-NN per channel output multiplier and shift.
  int32_t     *per_channel_output_multiplier;
  int32_t     *per_channel_output_shift;
};

inline float16_t normalize_fp16(float f)
{
  return (float16_t)std::min(std::max(f, SLI_MVP_FP16_MIN), SLI_MVP_FP16_MAX);
}

inline PaddingType RuntimePaddingType(TfLitePadding padding)
{
  switch (padding) {
    case TfLitePadding::kTfLitePaddingSame:
      return PaddingType::kSame;
    case TfLitePadding::kTfLitePaddingValid:
      return PaddingType::kValid;
    case TfLitePadding::kTfLitePaddingUnknown:
    default:
      return PaddingType::kNone;
  }
}

TfLiteStatus PopulateConvolutionQuantizationParams(
    TfLiteContext* context,
    const TfLiteTensor* input,
    const TfLiteTensor* filter,
    TfLiteTensor* output,
    const TfLiteFusedActivation& activation,
    int32_t* output_activation_min, int32_t* output_activation_max,
    float16_t* per_channel_scalers, int num_channels, float accumulator_multipler)
{
  auto affine_quantization =
        reinterpret_cast<const TfLiteAffineQuantization*>(filter->quantization.params);

  // Populate multiplier and shift using affine quantization.
  const float input_scale = input->params.scale;
  const float output_scale = output->params.scale;
  const float* filter_scales = affine_quantization->scale->data;

  for (int i = 0; i < num_channels; ++i) {
    // If per-tensor quantization parameter is specified, broadcast it along the
    // quantization dimension (channels_out).
    const float filter_scale = filter_scales[i];
    const float effective_output_scale = (input_scale * filter_scale) / output_scale;
    const float acc_output_scale = effective_output_scale * accumulator_multipler;
    per_channel_scalers[i] = normalize_fp16(acc_output_scale);
  }

  TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
          context, activation, output, output_activation_min,
          output_activation_max));

  return kTfLiteOk;
}

void *Init(TfLiteContext* context, const char* buffer, size_t length)
{
  (void)buffer;
  (void)length;
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node)
{
  int scratch_buffer_size = 0;

  TFLITE_DCHECK(node->user_data    != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  OpData* data = static_cast<OpData*>(node->user_data);
  const auto params = static_cast<const TfLiteConvParams*>(node->builtin_data);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TfLiteTensor* bias =
      micro_context->AllocateTempInputTensor(node, kBiasTensor);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TfLiteTensor* filter =
      micro_context->AllocateTempInputTensor(node, kFilterTensor);

  TF_LITE_ENSURE(context, input  != nullptr);
  TF_LITE_ENSURE(context, output != nullptr);
  TF_LITE_ENSURE(context, filter != nullptr);

  data->op_params.batches         = input->dims->data[0];
  data->op_params.in_channels     = input->dims->data[3];
  data->op_params.input_height    = input->dims->data[1];
  data->op_params.input_width     = input->dims->data[2];
  data->op_params.out_channels    = filter->dims->data[kConvQuantizedDimension];
  data->op_params.output_height   = output->dims->data[1];
  data->op_params.output_width    = output->dims->data[2];
  data->op_params.filter_height   = filter->dims->data[1];
  data->op_params.filter_width    = filter->dims->data[2];
  data->op_params.input_offset    = -input->params.zero_point;
  data->op_params.output_offset   = output->params.zero_point;
  data->op_params.stride_height   = params->stride_height;
  data->op_params.stride_width    = params->stride_width;
  data->op_params.dilation_height = params->dilation_height_factor;
  data->op_params.dilation_width  = params->dilation_width_factor;
  data->op_params.padding         = params->padding == kTfLitePaddingSame;

  int dummy_height, dummy_width;
  const auto padding = ComputePaddingHeightWidth(
                         params->stride_height, params->stride_width,
                         params->dilation_height_factor, params->dilation_width_factor,
                         data->op_params.input_height, data->op_params.input_width,
                         data->op_params.filter_height, data->op_params.filter_width,
                         params->padding,
                         &dummy_height, &dummy_width);

  data->op_params.pad_height = padding.height;
  data->op_params.pad_width  = padding.width;

  const int num_channels = data->op_params.out_channels;

  if (input->type == kTfLiteInt8) {
    if (sli_mvp_ml_conv2d_s8_is_supported(&data->op_params)) {
      data->supported = kMvp;

      float16_t *bias_data = static_cast<float16_t*>(context->AllocatePersistentBuffer(
                             context, num_channels * sizeof(float16_t)));
      if (bias != nullptr && bias->data.i32 != nullptr) {
        data->op_params.bias = bias_data;
        int32_t i32_bias;
        for (int i = 0; i < num_channels; i++) {
          i32_bias = bias->data.i32[i];
          bias_data[i] = float16_t(i32_bias * SLI_MVP_ACCUMULATOR_SCALER);
        }
      } else {
        data->op_params.bias = nullptr;
      }

      float16_t *scaler_data = static_cast<float16_t*>(context->AllocatePersistentBuffer(
                               context, num_channels * sizeof(float16_t)));
      data->op_params.output_scaler = scaler_data;

      scratch_buffer_size = sli_mvp_ml_conv2d_s8_get_scratch_buffer_size(&data->op_params);

      TF_LITE_ENSURE_STATUS(PopulateConvolutionQuantizationParams(
        context, input, filter, output, params->activation,
        reinterpret_cast<int32_t*>(&data->op_params.output_activation_min),
        reinterpret_cast<int32_t*>(&data->op_params.output_activation_max),
        scaler_data, num_channels, SLI_MVP_ACCUMULATOR_MULTIPLIER));

    } else {
      data->per_channel_output_multiplier = static_cast<int32_t*>(context->AllocatePersistentBuffer(
                                            context, num_channels * sizeof(int32_t)));
      data->per_channel_output_shift = static_cast<int32_t*>(context->AllocatePersistentBuffer(
                                       context, num_channels * sizeof(int32_t)));

      int32_t dummy_output_multiplier;
      int dummy_output_shift;
      TF_LITE_ENSURE_STATUS(tflite::PopulateConvolutionQuantizationParams(
        context, input, filter, bias, output, params->activation,
        &dummy_output_multiplier, &dummy_output_shift,
        reinterpret_cast<int32_t*>(&data->op_params.output_activation_min),
        reinterpret_cast<int32_t*>(&data->op_params.output_activation_max),
        data->per_channel_output_multiplier,
        reinterpret_cast<int32_t*>(data->per_channel_output_shift), num_channels));


      data->supported = kCmsisNN;
      cmsis_nn_conv_params       conv_params;
      conv_params.input_offset   = data->op_params.input_offset;
      conv_params.output_offset  = data->op_params.output_offset;
      conv_params.stride.h       = data->op_params.stride_height;
      conv_params.stride.w       = data->op_params.stride_width;
      conv_params.dilation.h     = data->op_params.dilation_height;
      conv_params.dilation.w     = data->op_params.dilation_width;
      conv_params.padding.h      = data->op_params.pad_height;
      conv_params.padding.w      = data->op_params.pad_width;
      conv_params.activation.min = data->op_params.output_activation_min;
      conv_params.activation.max = data->op_params.output_activation_max;

      cmsis_nn_dims input_dims;
      input_dims.n = data->op_params.batches;
      input_dims.h = data->op_params.input_height;
      input_dims.w = data->op_params.input_width;
      input_dims.c = data->op_params.in_channels;

      cmsis_nn_dims filter_dims;
      filter_dims.h = data->op_params.filter_height;
      filter_dims.w = data->op_params.filter_width;

      cmsis_nn_dims output_dims;
      output_dims.h = data->op_params.output_height;
      output_dims.w = data->op_params.output_width;
      output_dims.c = data->op_params.out_channels;

      scratch_buffer_size = arm_convolve_wrapper_s8_get_buffer_size(
                              &conv_params, &input_dims, &filter_dims, &output_dims);
#ifndef __arm__
        // If we're building for the wrapper
        // then just use the reference kernels
        // We still need the calculations above so we can
        // determine the required tensor arena size
        data->supported = kTFLMrefI8;
#endif // __arm__
    }

  } else if (input->type == kTfLiteFloat32) {
    data->supported = kTFLMrefF32;
    CalculateActivationRange(params->activation,
                             &data->activation_min_f32,
                             &data->activation_max_f32);

  } else {
    TF_LITE_KERNEL_LOG(context, "Type %s not currently supported.",
                       TfLiteTypeGetName(input->type));
    return kTfLiteError;
  }

  if(scratch_buffer_size > 0) {
    TF_LITE_ENSURE_STATUS(
      context->RequestScratchBufferInArena(
                 context, scratch_buffer_size, &data->scratch_buffer_index));
  } else {
    data->scratch_buffer_index = -1;
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);
  micro_context->DeallocateTempTfLiteTensor(output);
  if (bias != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(bias);
  }

  return kTfLiteOk;
}

TfLiteStatus eval_mvp_int8(TfLiteContext* context,
                           OpData* data,
                           const TfLiteEvalTensor* input,
                           const TfLiteEvalTensor* filter,
                           TfLiteEvalTensor* output)
{
  data->op_params.input  = tflite::micro::GetTensorData<int8_t>(input);
  data->op_params.output = tflite::micro::GetTensorData<int8_t>(output);
  data->op_params.filter = tflite::micro::GetTensorData<int8_t>(filter);

  // Add scratch buffer pointer to op_params
  if (data->scratch_buffer_index > -1){
    data->op_params.scratch_buffer = (float16_t*)context->GetScratchBuffer(context, data->scratch_buffer_index);
  }

  TF_LITE_ENSURE_EQ(context, SL_STATUS_OK, sli_mvp_ml_conv2d_s8(&data->op_params));

  return kTfLiteOk;
}

#ifdef __arm__
TfLiteStatus eval_cmsis_int8(TfLiteContext* context,
                             OpData* data,
                             const TfLiteEvalTensor* input,
                             const TfLiteEvalTensor* filter,
                             const TfLiteEvalTensor* bias,
                             TfLiteEvalTensor* output)
{
  cmsis_nn_dims input_dims;
  input_dims.n = data->op_params.batches;
  input_dims.h = data->op_params.input_height;
  input_dims.w = data->op_params.input_width;
  input_dims.c = data->op_params.in_channels;

  cmsis_nn_dims filter_dims;
  filter_dims.n = data->op_params.out_channels;
  filter_dims.h = data->op_params.filter_height;
  filter_dims.w = data->op_params.filter_width;
  filter_dims.c = data->op_params.in_channels;

  cmsis_nn_dims bias_dims;
  bias_dims.n = 1;
  bias_dims.h = 1;
  bias_dims.w = 1;
  bias_dims.c = data->op_params.out_channels;

  cmsis_nn_dims output_dims;
  output_dims.n = data->op_params.batches;
  output_dims.h = data->op_params.output_height;
  output_dims.w = data->op_params.output_width;
  output_dims.c = data->op_params.out_channels;

  cmsis_nn_per_channel_quant_params quant_params;
  quant_params.multiplier = data->per_channel_output_multiplier;
  quant_params.shift = data->per_channel_output_shift;

  cmsis_nn_conv_params       conv_params;
  conv_params.input_offset   = data->op_params.input_offset;
  conv_params.output_offset  = data->op_params.output_offset;
  conv_params.stride.h       = data->op_params.stride_height;
  conv_params.stride.w       = data->op_params.stride_width;
  conv_params.dilation.h     = data->op_params.dilation_height;
  conv_params.dilation.w     = data->op_params.dilation_width;
  conv_params.padding.h      = data->op_params.pad_height;
  conv_params.padding.w      = data->op_params.pad_width;
  conv_params.activation.min = data->op_params.output_activation_min;
  conv_params.activation.max = data->op_params.output_activation_max;

  cmsis_nn_context ctx;
  ctx.buf = nullptr;
  ctx.size = 0;

  if (data->scratch_buffer_index > -1) {
    ctx.buf = context->GetScratchBuffer(context, data->scratch_buffer_index);
  }
  TFLITE_DCHECK_EQ(ARM_CMSIS_NN_SUCCESS,
                   arm_convolve_wrapper_s8(
                     &ctx, &conv_params, &quant_params,
                     &input_dims,  tflite::micro::GetTensorData<int8_t>(input),
                     &filter_dims, tflite::micro::GetTensorData<int8_t>(filter),
                     &bias_dims,   bias == nullptr ? NULL : tflite::micro::GetTensorData<int32_t>(bias),
                     &output_dims, tflite::micro::GetTensorData<int8_t>(output)));

  return kTfLiteOk;
}
#endif // __arm__

TfLiteStatus eval_tflm_int8(OpData* data,
                            const TfLiteEvalTensor* input,
                            const TfLiteEvalTensor* filter,
                            const TfLiteEvalTensor* bias,
                            TfLiteEvalTensor* output)
{
  ConvParams op_params;

  op_params.input_offset             = data->op_params.input_offset;
  op_params.output_offset            = data->op_params.output_offset;
  op_params.stride_height            = data->op_params.stride_height;
  op_params.stride_width             = data->op_params.stride_width;
  op_params.dilation_height_factor   = data->op_params.dilation_height;
  op_params.dilation_width_factor    = data->op_params.dilation_width;
  op_params.padding_values.height    = data->op_params.pad_height;
  op_params.padding_values.width     = data->op_params.pad_width;
  op_params.quantized_activation_min = data->op_params.output_activation_min;
  op_params.quantized_activation_max = data->op_params.output_activation_max;

  reference_integer_ops::ConvPerChannel(
    op_params,
    data->per_channel_output_multiplier,
    data->per_channel_output_shift,
    tflite::micro::GetTensorShape(input),
    tflite::micro::GetTensorData<int8_t>(input),
    tflite::micro::GetTensorShape(filter),
    tflite::micro::GetTensorData<int8_t>(filter),
    tflite::micro::GetTensorShape(bias),
    bias == nullptr ? nullptr : tflite::micro::GetTensorData<int32_t>(bias),
    tflite::micro::GetTensorShape(output),
    tflite::micro::GetTensorData<int8_t>(output));

  return kTfLiteOk;
}

TfLiteStatus eval_float(TfLiteConvParams* params,
                        const OpData* data,
                        const TfLiteEvalTensor* input,
                        const TfLiteEvalTensor* filter,
                        const TfLiteEvalTensor* bias,
                        TfLiteEvalTensor* output)
{
  ConvParams op_params;
  op_params.padding_type           = RuntimePaddingType(params->padding);
  op_params.padding_values.width   = data->op_params.pad_width;
  op_params.padding_values.height  = data->op_params.pad_height;
  op_params.stride_width           = data->op_params.stride_width;
  op_params.stride_height          = data->op_params.stride_height;
  op_params.dilation_width_factor  = data->op_params.dilation_width;
  op_params.dilation_height_factor = data->op_params.dilation_height;
  op_params.float_activation_min   = data->activation_min_f32;
  op_params.float_activation_max   = data->activation_max_f32;

  reference_ops::Conv(op_params,
                      tflite::micro::GetTensorShape(input),
                      tflite::micro::GetTensorData<float>(input),
                      tflite::micro::GetTensorShape(filter),
                      tflite::micro::GetTensorData<float>(filter),
                      tflite::micro::GetTensorShape(bias),
                      bias == nullptr ? nullptr : tflite::micro::GetTensorData<float>(bias),
                      tflite::micro::GetTensorShape(output),
                      tflite::micro::GetTensorData<float>(output),
                      RuntimeShape(),
                      nullptr);
  return kTfLiteOk;
}

TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node)
{
  TfLiteStatus status = kTfLiteError;

  TFLITE_DCHECK(node->user_data    != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  auto* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);
  OpData* data = static_cast<OpData*>(node->user_data);

  const auto input  = tflite::micro::GetEvalInput(context, node, kInputTensor);
  const auto filter = tflite::micro::GetEvalInput(context, node, kFilterTensor);
  const auto bias   = NumInputs(node) == 3
                      ? tflite::micro::GetEvalInput(context, node, kBiasTensor)
                      : nullptr;
  auto output       = tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  if (data->supported == kMvp) {
    status = eval_mvp_int8(context, data, input, filter, output);
  }
#ifdef __arm__
  else if (data->supported == kCmsisNN) {
    status = eval_cmsis_int8(context, data, input, filter, bias, output);
  }
#endif
  else if (data->supported == kTFLMrefI8) {
    status = eval_tflm_int8(data, input, filter, bias, output);

  } else if (data->supported == kTFLMrefF32) {
    status = eval_float(params, data, input, filter, bias, output);
  }

  return status;
}

}  // namespace conv2d
}  // namespace sl

TFLMRegistration Register_CONV_2D() {
  return {/*init=*/sl::conv2d::Init,
          /*free=*/nullptr,
          /*prepare=*/sl::conv2d::Prepare,
          /*invoke=*/sl::conv2d::Invoke,
          /*reset=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr
  };
}

}  // namespace tflite
