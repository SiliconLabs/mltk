# API Reference


Once the MLTK is [installed](../installation.md) into the Python environment, it may be imported into a python script using:

```python
import mltk
```

Once the MLTK is imported, it's various APIs may be accessed.


**API Overview**

The following provides a general overview of the MLTK Python API:

| Name                                                                                                            | Description                                                                                                                  |
|-----------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| [Model Operations](https://siliconlabs.github.io/mltk/docs/python_api/operations/index.html)                    | Modeling operations such as profiling and training                                                                           |
| [MLTK Model](https://siliconlabs.github.io/mltk/docs/python_api/mltk_model/index.html)                          | Provides the root object of a [model specification](https://siliconlabs.github.io/mltk/docs/guides/model_specification.html) |
| [Tensorflow-Lite Model](https://siliconlabs.github.io/mltk/docs/python_api/tflite_model/index.html)             | Enables reading/writing `.tflite` model flatbuffer                                                                           |
| [Tensorflow-Lite Micro Model](https://siliconlabs.github.io/mltk/docs/python_api/tflite_micro_model/index.html) | Enables running `.tflite` models in the [Tensorflow-Lite Micro](https://github.com/tensorflow/tflite-micro) interpreter      |
| [Keras Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)                                        | The model object used by [Tensorflow](https://www.tensorflow.org/overview) during model training                             |
| [Data Preprocessing](https://siliconlabs.github.io/mltk/docs/python_api/data_preprocessing/index.html)          | Dataset preprocessing utilities                                                                                              |
| [Utilities](https://siliconlabs.github.io/mltk/docs/python_api/utils/index.html)                                | Common utilities                                                                                                             |
| [Reference Models](https://siliconlabs.github.io/mltk/docs/python_api/models/index.html)                        | Pre-trained reference models                                                                                                 |
| [Reference Datasets](https://siliconlabs.github.io/mltk/docs/python_api/datasets/index.html)                    | Datasets used by reference models                                                                                            |

**Package Directory Structure**

The MLTK Python package has the following structure:

| Name                                                                                                                   | Description                                                                                                                                                                                                                                                         |
|------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [mltk](https://github.com/siliconlabs/mltk/tree/master/mltk)                                                           | The root of the MLTK package                                                                                                                                                                                                                                        |
| [mltk.core](https://github.com/siliconlabs/mltk/tree/master/mltk/core)                                                 | Core modeling utilities, see the [Model Operations](https://siliconlabs.github.io/mltk/docs/python_api/operations/index.html) docs for more details                                                                                                                 |
| [mltk.core.model](https://github.com/siliconlabs/mltk/tree/master/mltk/core/model)                                     | Provides the root object of a [model specification](https://siliconlabs.github.io/mltk/docs/guides/model_specification.html), more details in the [MLTK Model](https://siliconlabs.github.io/mltk/docs/python_api/mltk_model/index.html) docs                       |
| [mltk.core.preprocess](https://github.com/siliconlabs/mltk/tree/master/mltk/core/preprocess)                           | Data pre-processing utilities, see the [Data Preprocessing](https://siliconlabs.github.io/mltk/docs/python_api/data_preprocessing/index.html) docs for more info                                                                                                    |
| [mltk.core.tflite_model](https://github.com/siliconlabs/mltk/tree/master/mltk/core/tflite_model)                       | Enables reading/writing `.tflite` model flatbuffers, more details in the [TfliteModel](https://siliconlabs.github.io/mltk/docs/python_api/tflite_model/index.html) docs                                                                                             |
| [mltk.core.tflite_model_parameters](https://github.com/siliconlabs/mltk/tree/master/mltk/core/tflite_model_parameters) | Enables read/writing custom parameters in a `.tflite` model flatbuffer                                                                                                                                                                                              |
| [mltk.core.tflite_micro](https://github.com/siliconlabs/mltk/tree/master/mltk/core/tflite_micro)                       | Enables running `.tflite` models in the [Tensorflow-Lite Micro](https://github.com/tensorflow/tflite-micro) interpreter, more details in the [Tensorflow-Lite Micro Wrapper](https://siliconlabs.github.io/mltk/docs/python_api/tflite_micro_model/index.html) docs |
| [mltk.core.keras](https://github.com/siliconlabs/mltk/tree/master/mltk/core/keras)                                     | Helper scripts for the [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras)                                                                                                                                                                             |
| [mltk.utils](https://github.com/siliconlabs/mltk/tree/master/mltk/utils)                                               | Common utility scripts, more details in the [utilities](https://siliconlabs.github.io/mltk/docs/python_api/utils/index.html) docs                                                                                                                                   |
| [mltk.cli](https://github.com/siliconlabs/mltk/tree/master/mltk/cli)                                                   | MLTK Command-Line Interface (CLI) scripts                                                                                                                                                                                                                           |
| [mltk.models](https://github.com/siliconlabs/mltk/tree/master/mltk/models)                                             | Reference models, more details in the [Reference models](https://siliconlabs.github.io/mltk/docs/python_api/models/index.html) docs                                                                                                                                 |
| [mltk.datasets](https://github.com/siliconlabs/mltk/tree/master/mltk/datasets)                                         | Reference datasets, more details in the [Reference datasets](https://siliconlabs.github.io/mltk/docs/python_api/datasets/index.html) docs                                                                                                                           |






```{eval-rst}
.. toctree::
   :maxdepth: 1
   :hidden:

   ./operations/index
   ./mltk_model/index
   ./tflite_micro_model/index
   ./tflite_model/index
   ./keras_model
   ./data_preprocessing/index
   ./utils/index
```