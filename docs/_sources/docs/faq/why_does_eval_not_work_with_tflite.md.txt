# Why does the Keras (.h5) model work during evaluation but the TF-Lite (.tflite) does not?

If the Keras (`.h5`) model gets acceptable [evaluation](../guides/model_evaluation.md) results
but the TF-Lite (`.tflite`) does not, then this could mean that the input data is not being properly
converted.

The easiest fix is to change your `.tflite` model's input data type to `float`, e.g.:

```python
my_model.tflite_converter['inference_input_type'] = tf.float32
my_model.tflite_converter['inference_output_type'] = tf.float32
```
Otherwise, you must ensure that your data is in a format the the `.tflite` model expects.

Refer to the FAQ [Why is the model not returning correct results on the embedded device?](./why_is_model_not_working_on_embedded.md)
for more details on how to use the `int8` data type.


