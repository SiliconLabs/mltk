# Model Object Types


The following model objects are used by the MLTK APIs:

- [MltkModel](./mltk_model.md) - Model object defined by a [model specification](../../guides/model_specification.md)
- [TfliteModel](./tflite_model.md) - Provides access to a ``.tflite`` model file
- [TfliteMicroModel](./tflite_micro_model.md) - Provides access to the [Tensorflow-Lite Micro Interpreter](https://github.com/tensorflow/tflite-micro)
- [KerasModel](./keras_model.md) - The actual machine learning model which is used by Tensorflow during training



```{toctree}
:maxdepth: 1
:hidden:

./mltk_model
./tflite_model
./tflite_micro_model
./keras_model

```
