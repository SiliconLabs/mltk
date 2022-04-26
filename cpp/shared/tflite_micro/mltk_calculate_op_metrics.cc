



#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

#include "mltk_tflite_micro_internal.hpp"





using namespace tflite::micro;


namespace mltk
{


#define GetInput(index) tflite::micro::GetEvalInput(context, &node_and_registration.node, index)
#define GetOutput(index) tflite::micro::GetEvalOutput(context, &node_and_registration.node, index)



/*************************************************************************************************/
static uint32_t add_activation_ops(TfLiteFusedActivation activation, int count)
{
    uint32_t retval = 0;

    switch(activation)
    {
    case kTfLiteActNone:
        break;
    case kTfLiteActRelu:
    case kTfLiteActReluN1To1:
    case kTfLiteActRelu6:
        retval = count * 2; // min(max(lower, x), upper)
        break;
    default:
        break;
    }

    return retval;
}

/*************************************************************************************************/
static void calculate_fully_connected(const TfLiteContext* context,
                                      const tflite::NodeAndRegistration& node_and_registration,
                                      profiling::Metrics& metrics)
{
    constexpr int kWeightsTensor = 1;
    constexpr int kOutputTensor = 0;
    constexpr int kBiasTensor = 2;

    const auto& node = node_and_registration.node;
    const auto params = reinterpret_cast<TfLiteFullyConnectedParams*>(node.builtin_data);
    const auto output = GetOutput(kOutputTensor);
    const auto weights = GetInput(kWeightsTensor);
    const auto bias = GetInput(kBiasTensor);

    const auto weights_shape = GetTensorShape(weights);
    const auto output_shape =  GetTensorShape(output);

    const int output_depth = output_shape.Dims(output_shape.DimensionsCount() - 1);
    const int accum_depth = weights_shape.Dims(weights_shape.DimensionsCount() - 1);

    metrics.macs = output_depth * accum_depth;
    metrics.ops = metrics.macs * 2;
    if(bias != nullptr)
    {
        metrics.ops += output_depth;
    }
    metrics.ops += add_activation_ops(params->activation, output_depth);
}

/*************************************************************************************************/
static void calculate_conv2d(const TfLiteContext* context,
                             const tflite::NodeAndRegistration& node_and_registration,
                             profiling::Metrics& metrics)
{
    constexpr int kInputTensor = 0;
    constexpr int kFilterTensor = 1;
    constexpr int kOutputTensor = 0;
    constexpr int kBiasTensor = 2;


    const auto& node = node_and_registration.node;
    const auto params = reinterpret_cast<TfLiteConvParams*>(node.builtin_data);
    const auto output = GetOutput(kOutputTensor);
    const auto input = GetInput(kInputTensor);
    const auto filter = GetInput(kFilterTensor);
    const auto bias = GetInput(kBiasTensor);

    const auto input_shape = GetTensorShape(input);
    const auto filter_shape = GetTensorShape(filter);
    const auto output_shape = GetTensorShape(output);

    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int input_depth = input_shape.Dims(3);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);
    const int output_depth = output_shape.Dims(3);

    metrics.macs = ((filter_height * filter_width * input_depth) * output_width * output_height * output_depth);
    metrics.ops = metrics.macs * 2;
    if(bias != nullptr)
    {
        metrics.ops += output_width * output_height * output_depth;
    }

    metrics.ops += add_activation_ops(params->activation, output_width * output_height * output_depth);
}

/*************************************************************************************************/
static void calculate_transpose_conv(const TfLiteContext* context,
                                     const tflite::NodeAndRegistration& node_and_registration,
                                     profiling::Metrics& metrics)
{
    //constexpr int kOutputShapeTensor = 0;
    constexpr int kFilterTensor = 1;
    constexpr int kInputTensor = 2;
    constexpr int kBiasTensor = 3;
    constexpr int kOutputTensor = 0;


    const auto& node = node_and_registration.node;
    const auto params = reinterpret_cast<TfLiteTransposeConvParams*>(node.builtin_data);
    const auto output = GetOutput(kOutputTensor);
    const auto input = GetInput(kInputTensor);
    const auto filter = GetInput(kFilterTensor);
    const auto bias = GetInput(kBiasTensor);

    const auto input_shape = GetTensorShape(input);
    const auto filter_shape = GetTensorShape(filter);
    const auto output_shape = GetTensorShape(output);

    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int input_depth = input_shape.Dims(3);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int output_depth = output_shape.Dims(3);

    int output_width, output_height;
    tflite::ComputePaddingHeightWidth(
        params->stride_height, params->stride_width,
        1, 1, input_height,
        input_width, filter_height, filter_width, params->padding, &output_height, &output_width);

    metrics.macs = ((filter_height * filter_width * output_depth) * input_width * input_height * input_depth);
    metrics.ops = metrics.macs * 2;
    if(bias != nullptr)
    {
        metrics.ops += output_width * output_height * output_depth;
    }
}

/*************************************************************************************************/
static void calculate_depthwise_conv2d(const TfLiteContext* context,
                                       const tflite::NodeAndRegistration& node_and_registration,
                                        profiling::Metrics& metrics)
{
    constexpr int kInputTensor = 0;
    constexpr int kOutputTensor = 0;
    constexpr int kFilterTensor = 1;
    constexpr int kBiasTensor = 2;


    const auto& node = node_and_registration.node;
    const auto params = reinterpret_cast<TfLiteDepthwiseConvParams*>(node.builtin_data);
    const auto input = GetInput(kInputTensor);
    const auto output = GetOutput(kOutputTensor);
    const auto filter = GetInput(kFilterTensor);
    const auto bias = GetInput(kBiasTensor);

    const auto output_shape = GetTensorShape(output);
    const auto input_shape = GetTensorShape(input);
    const auto filter_shape = GetTensorShape(filter);

    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int input_depth = input_shape.Dims(3);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int depth_multiplier = params->depth_multiplier;
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);

    metrics.macs = (filter_width * filter_height * depth_multiplier * input_depth * output_width * output_height);
    metrics.ops = metrics.macs * 2;
    if(bias != nullptr)
    {
        metrics.ops += depth_multiplier * input_depth * output_width * output_height;
    }
    metrics.ops += add_activation_ops(params->activation, depth_multiplier * input_depth * output_width * output_height);
}

/*************************************************************************************************/
static void calculate_max_pool2d(const TfLiteContext* context,
                                 const tflite::NodeAndRegistration& node_and_registration,
                                 profiling::Metrics& metrics)
{
    constexpr int kInputTensor = 0;
    constexpr int kOutputTensor = 0;

    const auto& node = node_and_registration.node;
    const auto params = reinterpret_cast<TfLitePoolParams*>(node.builtin_data);

    const auto output = GetOutput(kOutputTensor);
    const auto input = GetInput(kInputTensor);
    const auto input_shape = GetTensorShape(input);
    const auto output_shape = GetTensorShape(output);

    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);
    const int output_depth = output_shape.Dims(3);

    metrics.ops = (output_height * output_width * output_depth * (params->filter_width * params->filter_height));
    metrics.ops += add_activation_ops(params->activation, output_height * output_width * output_depth);
}

/*************************************************************************************************/
static void calculate_average_pool2d(const TfLiteContext* context,
                                     const tflite::NodeAndRegistration& node_and_registration,
                                     profiling::Metrics& metrics)
{
    constexpr int kInputTensor = 0;
    constexpr int kOutputTensor = 0;

    const auto& node = node_and_registration.node;
    const auto params = reinterpret_cast<TfLitePoolParams*>(node.builtin_data);

    const auto output = GetOutput(kOutputTensor);
    const auto input = GetInput(kInputTensor);
    const auto input_shape = GetTensorShape(input);
    const auto output_shape = GetTensorShape(output);

    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);
    const int output_depth = output_shape.Dims(3);

    metrics.ops = (output_height * output_width * output_depth * (params->filter_width * params->filter_height + 1));
    metrics.ops += add_activation_ops(params->activation, output_height * output_width * output_depth);
}

/*************************************************************************************************/
static void calculate_softmax(const TfLiteContext* context,
                              const tflite::NodeAndRegistration& node_and_registration,
                              profiling::Metrics& metrics)
{
    const auto input = GetInput(0);
    const auto& dims = *input->dims;
    const int input_size = dims.data[(dims.size == 2) ? 1 : 0];

    metrics.ops = input_size * (1 + 3 + 1);
}

/*************************************************************************************************/
static void calculate_add(const TfLiteContext* context,
                           const tflite::NodeAndRegistration& node_and_registration,
                           profiling::Metrics& metrics)
{
    const auto input = GetInput(0);
    const auto input_shape = tflite::micro::GetTensorShape(input);
    const int flat_size = input_shape.FlatSize();

    metrics.ops = flat_size;
}

/*************************************************************************************************/
static void calculate_quantize(const TfLiteContext* context,
                           const tflite::NodeAndRegistration& node_and_registration,
                           profiling::Metrics& metrics)
{
    const auto input = GetInput(0);
    const auto input_shape = tflite::micro::GetTensorShape(input);
    const int flat_size = input_shape.FlatSize();

    // https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/kernels/internal/reference/quantize.h#L31
    // 1 divide
    // 1 add
    // 1 min
    // 1 max
    metrics.ops = flat_size * 4;
}

/*************************************************************************************************/
static void calculate_dequantize(const TfLiteContext* context,
                           const tflite::NodeAndRegistration& node_and_registration,
                           profiling::Metrics& metrics)
{
    const auto input = GetInput(0);
    const auto input_shape = tflite::micro::GetTensorShape(input);
    const int flat_size = input_shape.FlatSize();

    // https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/kernels/internal/reference/dequantize.h
    // 1 multiply
    // 1 subtract
    metrics.ops = flat_size * 2;
}

/*************************************************************************************************/
static void calculate_pad(const TfLiteContext* context,
                           const tflite::NodeAndRegistration& node_and_registration,
                           profiling::Metrics& metrics)
{
    const auto output = GetOutput(0);
    const auto output_shape = tflite::micro::GetTensorShape(output);
    const int flat_size = output_shape.FlatSize();

    // https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/kernels/internal/reference/pad.h
    // 1 mv
    // 10/2 =  5 comparisions
    metrics.ops = flat_size * 6;
}

/*************************************************************************************************/
static void calculate_reshape(const TfLiteContext* context,
                           const tflite::NodeAndRegistration& node_and_registration,
                           profiling::Metrics& metrics)
{
    const auto input = GetInput(0);
    const auto output = GetOutput(0);
    const auto input_shape = tflite::micro::GetTensorShape(input);
    const int flat_size = input_shape.FlatSize();

    // If a memcpy is required
    // then include 1 op for each input element
    // https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/kernels/reshape.cc#L94
    if(input->data.raw != output->data.raw) 
    {
        metrics.ops = flat_size;
    }
}


/*************************************************************************************************/
static void calculate_mean(const TfLiteContext* context,
                           const tflite::NodeAndRegistration& node_and_registration,
                           profiling::Metrics& metrics)
{
//    const auto input = GetInput(0);
//    const auto output = GetOutput0);
//    const auto input_shape = GetTensorShape(input);
//    const auto output_shape = GetTensorShape(output);
//
//    const int input_height = input_shape.Dims(1);
//    const int input_width = input_shape.Dims(2);
//    const int output_depth = output_shape.Dims(3);

    // FIXME: figure out why this calculation is returning a bogus value
    metrics.ops = 0;// (output_depth * input_height * input_width) + output_depth;
}

/*************************************************************************************************/
static void calculate_resize_nearest_neighbor(const TfLiteContext* context,
                                              const tflite::NodeAndRegistration& node_and_registration,
                                              profiling::Metrics& metrics)
{
    constexpr int kSizeTensor = 1;
    constexpr int GetNearestNeighbor_flops = 8; // just an approximation for the number of operations in GetNearestNeighbor()

    const auto size = GetInput(kSizeTensor);
    const auto size_data =  GetTensorData<int32_t>(size);

    const int32_t output_height = size_data[0];
    const int32_t output_width = size_data[1];

    metrics.ops = output_height * (GetNearestNeighbor_flops + output_width * GetNearestNeighbor_flops);
}



/*************************************************************************************************/
static void calculate_relu(const TfLiteContext* context,
                           const tflite::NodeAndRegistration& node_and_registration,
                           profiling::Metrics& metrics)
{
    constexpr int kInputTensor = 0;

    const auto input = GetInput(0);
    const auto input_shape = tflite::micro::GetTensorShape(input);

    metrics.ops = input_shape.FlatSize() * 2; // 2 ops, one for matching upper and lower limits
}




/*************************************************************************************************/
bool calculate_op_metrics(const TfLiteContext* context,
                          const tflite::NodeAndRegistration& node_and_registration,
                          profiling::Metrics& metrics)
{
    const auto builtin_code = node_and_registration.registration->builtin_code;

    if(builtin_code == tflite::BuiltinOperator_FULLY_CONNECTED)
    {
        calculate_fully_connected(context, node_and_registration, metrics);
    }
    else if(builtin_code == tflite::BuiltinOperator_CONV_2D)
    {
        calculate_conv2d(context, node_and_registration, metrics);
    }
    else if(builtin_code == tflite::BuiltinOperator_DEPTHWISE_CONV_2D)
    {
        calculate_depthwise_conv2d(context, node_and_registration, metrics);
    }
    else if(builtin_code == tflite::BuiltinOperator_TRANSPOSE_CONV)
    {
        calculate_transpose_conv(context, node_and_registration, metrics);
    }
    else if(builtin_code == tflite::BuiltinOperator_MAX_POOL_2D)
    {
        calculate_max_pool2d(context, node_and_registration, metrics);
    }
    else if(builtin_code == tflite::BuiltinOperator_AVERAGE_POOL_2D)
    {
        calculate_average_pool2d(context, node_and_registration, metrics);
    }
    else if(builtin_code == tflite::BuiltinOperator_SOFTMAX)
    {
        calculate_softmax(context, node_and_registration, metrics);
    }
    else if(builtin_code == tflite::BuiltinOperator_ADD)
    {
        calculate_add(context, node_and_registration, metrics);
    }
    else if(builtin_code == tflite::BuiltinOperator_MEAN)
    {
        calculate_mean(context, node_and_registration, metrics);
    }
    else if(builtin_code == tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR)
    {
        calculate_resize_nearest_neighbor(context, node_and_registration, metrics);
    }
    else if(builtin_code == tflite::BuiltinOperator_RELU || builtin_code == tflite::BuiltinOperator_RELU6)
    {
        calculate_relu(context, node_and_registration, metrics);
    }
    else if(builtin_code == tflite::BuiltinOperator_QUANTIZE)
    {
        calculate_quantize(context, node_and_registration, metrics);
    }
    else if(builtin_code == tflite::BuiltinOperator_DEQUANTIZE)
    {
        calculate_dequantize(context, node_and_registration, metrics);
    }
    else if(builtin_code == tflite::BuiltinOperator_PAD)
    {
        calculate_pad(context, node_and_registration, metrics);
    }
    else if(builtin_code == tflite::BuiltinOperator_RESHAPE)
    {
        calculate_reshape(context, node_and_registration, metrics);
    }
    else if(builtin_code == tflite::BuiltinOperator_SHAPE ||
            builtin_code == tflite::BuiltinOperator_STRIDED_SLICE ||
            builtin_code == tflite::BuiltinOperator_PACK ||
            builtin_code == tflite::BuiltinOperator_SPLIT ||
            builtin_code == tflite::BuiltinOperator_EXPAND_DIMS ||
            builtin_code == tflite::BuiltinOperator_CONCATENATION)
    {
        // ignore these layers for now
    }
    else
    {
        return false;
    }

    return true;
}


} // namespace mltk
