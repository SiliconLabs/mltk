# How can I reduce my model's size?

A model's size requirements are largely dependent on the application, however,
there are several things to keep in mind which can help to reduce its
resource requirements.

## What are the model resource requirements?

The resources required by a machine learning model are:  
- __RAM__ - To hold the working memory
- __Flash__ - To hold the trained model parameters (i.e. weights and filters)
- __Processing cycles__ - The CPU and/or accelerator cycles needed to execute the model, this directly determines how long it takes to execute the model

These values can be determined using the [Model Profiler](../guides/model_profiler.md).

Additionally, the model profiler will indicate if a layer is not able to be accelerated
by a hardware accelerator, such as the MVP. When this happens the layer is executed by
a slower software implementation. In this case, the layer can be reduced using the following
tips so that it fits on the hardware accelerator, and thus executes faster.

## Reduce model input size

Reducing the size of the model input tensor(s) is usually one of the 
most effective ways of reducing the model's total resource requirements.

## Use int8 model input data type

`float32` is a common model input data type.
While this is useful for automatically quantizing
the raw input data, it can increase the model's RAM
usage by ~4x compared to the `int8` input data type.

## Reduce Filter Count

The [Conv2D](https://keras.io/api/layers/convolution_layers/convolution2d/) layer
has a `filters` parameter and the [DepthwiseConv2D](https://keras.io/api/layers/convolution_layers/depthwise_convolution2d/) layer 
has a `depth_multiplier` parameter.

These can increase the model's accuracy at the expense of additional processing and thus execution latency.
Reducing these values can reduce the number of model operations and thus execution time.

## Increase Strides

The [Conv2D](https://keras.io/api/layers/convolution_layers/convolution2d/) and
[DepthwiseConv2D](https://keras.io/api/layers/convolution_layers/depthwise_convolution2d/) layers
have a `strides` parameter. The smaller this value is the more computations that are required 
as well as the larger the layer's output size.

Increasing this value can reduce the number of model operations and thus execution time.
Increasing this value also reduce the layer's output size which can help to make the layer fit
within the hardware accelerator's constraints.

## Decrease kernel sizes

The [Conv2D](https://keras.io/api/layers/convolution_layers/convolution2d/) and
[DepthwiseConv2D](https://keras.io/api/layers/convolution_layers/depthwise_convolution2d/) layers
have a `kernel_size` parameter. The larger this value is the more computations that are required.

Reducing this value can reduce the number of model operations and thus execution time.

## Reduce FullyConnected units

The [Dense](https://keras.io/api/layers/core_layers/dense/) (a.k.a. FullyConnected) layer
has a `units` parameter. Increasing this value can increase model accuracy at the expense
of additional computations.

Decreasing this value can reduce the number of model operations and thus execution time.

## Use pooling layers

The [MaxPool2D](https://keras.io/api/layers/pooling_layers/max_pooling2d/) and 
[AveragePool2D](https://keras.io/api/layers/pooling_layers/average_pooling2d/) layers
effectively down sample the previous layer. This can help to reduce the number of
model operations and thus execution time.