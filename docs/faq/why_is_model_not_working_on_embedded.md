# Why is the model not returning correct results on the embedded device?

There can be many reasons why the output of the  `.tflite` model does not generate valid results
when executing on an embedded device.

Some of these reasons include:

## Input Data Preprocessing

One of the more common reasons is that the input data is not formatted correctly.
Recall that whatever preprocessing is done to the dataset during model training _must_
also be done to the input samples on the embedded device at runtime. So, for instance,
if the training dataset is comprised of spectrograms, then whatever algorithms were used
to convert the raw audio samples into the spectrograms must also be used on the embedded device
(See the [AudioFeatureGenerator](../audio/audio_feature_generator.md) for how the MLTK can aid spectrogram generation).

The MLTK also supports creating [Python Wrappers](../cpp_development/wrappers/index.md) which allows for sharing C++ source code
between the model training scripts (i.e. Python) and the embedded device (i.e. C++). With this, algorithms can
be developed in C++ and used to preprocess the data during model training. Later, the _exact_ same C++ algorithms
can be built into the embedded firmware application. This way, the preprocessing algorithms only need to be written once
and can be shared between model training and model execution.

## Input Data Type

Another common issue can occur when the `.tflite` model input has an `int8` data type, e.g.:

```python
my_model.tflite_converter['inference_input_type'] = tf.int8
my_model.tflite_converter['inference_output_type'] = tf.int8
```

but the raw data uses another data type, e.g. `uint8`.

In this case, both the model training scripts _and_ embedded device must
convert the sample data to `int8`.

For example, say we're creating an image classification model and our dataset contains `uint8` images.
But, we want our model's input data type to be `int8`.

The our [model specification script](../guides/model_specification.md) might contain:

```python
# Tell the TF-Lite Converter to use int8 model input/output data types
my_model.tflite_converter['inference_input_type'] = tf.int8
my_model.tflite_converter['inference_output_type'] = tf.int8

...

# This is called by the ParallelImageDataGenerator() for each training sample
# It converts the data type from uint8 to int8
def convert_img_from_uint8_to_int8(params:ParallelProcessParams, x:np.ndarray) -> np.ndarray:
  # x is a float32 dtype but has an uint8 range
  x = np.clip(x, 0, 255) # The data should already been in the uint8 range, but clip it just to be sure
  x = x - 128 # Convert from uint8 to int8
  x = x.astype(np.int8)
  return x

# Define the data generator with the data conversion callback
my_model.datagen = ParallelImageDataGenerator(
  preprocessing_function=convert_img_from_uint8_to_int8,
  ...
```

With this, the model is trained with `int8` input data samples.


__Additionally__, on the embedded device, we must manually convert
the `uint8` data from the camera to `int8`, e.g.:

```c++
for(int i = 0; i < image_length; ++i)
{
  model_input->data.int8[i] = (int8_t)(image_data[i] - 128);
}
```

## Hint: Just use float32

You can skip all of the above by using a `float32` input data type, e.g.:

```python
my_model.tflite_converter['inference_input_type'] = tf.float32
my_model.tflite_converter['inference_output_type'] = tf.float32
```

With this, this is no need for the `convert_img_from_uint8_to_int8()` callback during training
nor the `image_data[i] - 128` on the embedded device.  
The raw `uint8` image data can be directly used during training _and_ on the embedded device.  
(However, on the embedded device, you'll need to convert the image data from `uint8` to `float`).

This works because the [TfliteConverter](https://www.tensorflow.org/lite/convert) automatically 
adds `Quantize` and `Dequantize` layers to the `.tflite` which internally convert the `float` input data to/from `int8`.

Using `float32` as the model input data type is useful as the conversion is automatically handled by the `.tflite` model.
However, it does require additional RAM and processing cycles.

It requires additional RAM because the input tensor buffer increases by 4x (i.e.`sizeof(int8)` vs `sizeof(float)`).  
Also, additional cycles are required to convert to/from `int8` and `float`.

