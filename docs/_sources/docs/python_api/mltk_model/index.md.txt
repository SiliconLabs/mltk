# MLTK Model

The [MltkModel](mltk.core.MltkModel) is the root object used to create a [model specification](../../guides/model_specification.md).

The [model specification](../../guides/model_specification.md) should define an object that inherits [MltkModel](mltk.core.MltkModel)
and any other required mixins. Once the object is defined and instantiated, the various properties should be populated.
After the model specification is finished, it may be invoked with one of the various [Model Operations](../operations/index.md).

## Example Usage


The following is a snippet from the [basic_example](https://siliconlabs.github.io/mltk/docs/python_api/models/examples/basic_example.html) reference model:


```python
import mltk.core as mltk_core

class MyModel(
    mltk_core.MltkModel,    # We must inherit the MltkModel class
    mltk_core.TrainMixin,   # We also inherit the TrainMixin since we want to train this model
    mltk_core.DatasetMixin, # We also need the DatasetMixin mixin to provide the relevant dataset properties
    mltk_core.EvaluateClassifierMixin,  # While not required, also inherit EvaluateClassifierMixin to help will generating evaluation for our classification model
):
    pass

my_model = MyModel()

my_model.version = 1
my_model.description = 'Basic model specification example'
my_model.classes = ['cat', 'dog', 'goat']
my_model.class_weights = 'balanced'
my_model.batch_size = 32
my_model.epochs = 100
my_model.validation_split = 0.2
...

if __name__ == '__main__':
    # Train the model
    # This does the same as issuing the command: mltk train basic_example
    mltk_core.train_model(my_model, clean=True)
    # Evaluate the model against the quantized .tflite (i.e. int8) model
    # This does the same as issuing the command: mltk evaluate basic_example --tflite
    mltk_core.evaluate_model(my_model, tflite=True)
    # Profile the model in the simulator
    # This does the same as issuing the command: mltk profile basic_example
    mltk_core.profile_model(my_model)

```

See the [reference models](https://siliconlabs.github.io/mltk/docs/python_api/models/index.html) for more examples.

Additional [Model Utilities](https://siliconlabs.github.io/mltk/docs/python_api/mltk_model/utilities.html) are also available.

## API Reference


The following [MltkModel](mltk.core.MltkModel) mixins are available:


```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: model
   :template: custom-class-template.rst

   mltk.core.MltkModel

.. autosummary::
   :nosignatures:
   :toctree: train_mixin
   :template: custom-class-template.rst

   mltk.core.TrainMixin

.. autosummary::
   :nosignatures:
   :toctree: dataset_mixin
   :template: custom-class-template.rst

   mltk.core.DatasetMixin

.. autosummary::
   :nosignatures:
   :toctree: audio_dataset_mixin
   :template: custom-class-template.rst

   mltk.core.AudioDatasetMixin

.. autosummary::
   :nosignatures:
   :toctree: image_dataset_mixin
   :template: custom-class-template.rst

   mltk.core.ImageDatasetMixin

.. autosummary::
   :nosignatures:
   :toctree: evaluate_mixin
   :template: custom-class-template.rst

   mltk.core.EvaluateMixin

.. autosummary::
   :nosignatures:
   :toctree: evaluate_autoencoder_mixin
   :template: custom-class-template.rst

   mltk.core.EvaluateAutoEncoderMixin

.. autosummary::
   :nosignatures:
   :toctree: evaluate_classifier_mixin
   :template: custom-class-template.rst

   mltk.core.EvaluateClassifierMixin

.. autosummary::
   :nosignatures:
   :toctree: ssh_mixin
   :template: custom-class-template.rst

   mltk.core.SshMixin

.. autosummary::
   :nosignatures:
   :toctree: weights_and_biases_mixin
   :template: custom-class-template.rst

   mltk.core.WeightsAndBiasesMixin

.. autosummary::
   :nosignatures:
   :toctree: mltk_dataset
   :template: custom-class-template.rst

   mltk.core.MltkDataset

.. autosummary::
   :nosignatures:
   :toctree: model_event
   :template: custom-class-template.rst

   mltk.core.MltkModelEvent

```


```{toctree}
:maxdepth: 2
:hidden:

./model
./train_mixin
./dataset_mixin
./image_dataset_mixin
./audio_dataset_mixin
./evaluate_mixin
./evaluate_classifier_mixin
./evaluate_autoencoder_mixin
./ssh_mixin
./weights_and_biases_mixin
./utilities
./mltk_dataset
./model_event
```