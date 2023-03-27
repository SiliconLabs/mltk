import enum


class MltkModelEvent(str, enum.Enum):
    """Events that are triggered at various stages of :py:class:`~MltkModel` execution.

    See :py:class:`~MltkModel.add_event_handler` for more details.
    """
    def _generate_next_value_(name: str, start: int, count: int, last_values: list) -> str: #pylint: disable=no-self-argument
        return name


    BEFORE_MODEL_LOAD = enum.auto()
    """Invoked **before** the :py:class:`~MltkModel` is fully loaded.

    This event does not have any additional keyword arguments.
    """

    AFTER_MODEL_LOAD = enum.auto()
    """Invoked **after** the :py:class:`~MltkModel` is fully loaded.

    This event does not have any additional keyword arguments.
    """

    BEFORE_LOAD_DATASET = enum.auto()
    """Invoked at the beginning of :py:class:`~DatasetMixin.load_dataset`

    This has the additional keyword arguments:

    - **subset** - One of training, validation or evaluation
    - **test** - True if the data is being loaded for testing
    """

    AFTER_LOAD_DATASET = enum.auto()
    """Invoked at the end of :py:class:`~DatasetMixin.load_dataset`

    This has the additional keyword arguments:

    - **subset** - One of training, validation or evaluation
    - **test** - True if the data is being loaded for testing
    """

    BEFORE_UNLOAD_DATASET = enum.auto()
    """Invoked at the beginning of :py:class:`~DatasetMixin.unload_dataset`

    This event does not have any additional keyword arguments.
    """

    AFTER_UNLOAD_DATASET = enum.auto()
    """Invoked at the end of :py:class:`~DatasetMixin.unload_dataset`

    This event does not have any additional keyword arguments.
    """

    SUMMARIZE_DATASET = enum.auto()
    """Invoked at the end of :py:class:`~DatasetMixin.summarize_dataset`

    This has the additional keyword arguments:

    - **summary** - The generated summary as a string, the summary cannot be modified in the event handler
    - **summary_dict** - The generated summary as ``summary_dict=dict(value=summary)``, ``summary_dict['value']`` may be modified in the event handler
    """

    SUMMARIZE_MODEL = enum.auto()
    """Invoked at the end of :py:class:`~summarize_model`

    This has the additional keyword arguments:

    - **summary** - The generated summary as a string, the summary cannot be modified in the event handler
    - **summary_dict** - The generated summary as ``summary_dict=dict(value=summary)``, Update ``summary_dict['value']`` to return a new summary by the event handler
    """

    TRAIN_STARTUP = enum.auto()
    """Invoked at the beginning of :py:class:`~train_model`

    This has the additional keyword arguments:

    - **post_process** - True if post-processing is enabled
    """

    BEFORE_BUILD_TRAIN_MODEL = enum.auto()
    """Invoked before :py:class:`~TrainMixin.build_model_function` is called

    This event does not have any additional keyword arguments.
    """

    AFTER_BUILD_TRAIN_MODEL = enum.auto()
    """Invoked after :py:class:`~TrainMixin.build_model_function` is called

    This has the additional keyword arguments:

    - **keras_model** - The built Keras model
    """

    POPULATE_TRAIN_CALLBACKS = enum.auto()
    """Invoked during :py:class:`~train_model` before Keras training starts.

    This has the additional keyword arguments:

    - **keras_callbacks** - A list of Keras Callbacks that will be passed to `KerasModel.fit() <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_
    """

    BEFORE_TRAIN = enum.auto()
    """Invoked during :py:class:`~train_model` before Keras training

    This has the additional keyword arguments:

    - **fit_kwargs** - Keyword args passed to `KerasModel.fit() <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_
    """

    AFTER_TRAIN = enum.auto()
    """Invoked during :py:class:`~train_model` after Keras training

    This has the additional keyword arguments:

    - **training_history** - The value returned by `KerasModel.fit() <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_
    """

    BEFORE_SAVE_TRAIN_MODEL = enum.auto()
    """Invoked during :py:class:`~train_model` before the trained model is saved

    This has the additional keyword arguments:

    - **keras_model** - The trained Keras model, this cannot be modified by the event handler
    - **keras_model_dict** - The trained Keras model as ``keras_model_dict=dict(value=keas_model)``, update ``keras_model_dict['value']`` to return a new model by the event handler
    """

    AFTER_SAVE_TRAIN_MODEL = enum.auto()
    """Invoked during :py:class:`~train_model` after the trained model is saved

    This has the additional keyword arguments:

    - **keras_model** - The trained Keras model, this cannot be modified by the event handler
    - **keras_model_dict** - The trained Keras model as ``keras_model_dict=dict(value=keas_model)``, update ``keras_model_dict['value']`` to return a new model by the event handler
    """

    BEFORE_SAVE_TRAIN_RESULTS = enum.auto()
    """Invoked during :py:class:`~train_model` before the training results are saved

    This has the additional keyword arguments:

    - **keras_model** - The trained Keras model, this cannot be modified by the event handler
    - **results** - The model :py:class:`~TrainingResults`
    - **output_dir** - Directory path where the results are saved
    """

    AFTER_SAVE_TRAIN_RESULTS = enum.auto()
    """Invoked during :py:class:`~train_model` after the training results are saved

    This has the additional keyword arguments:

    - **keras_model** - The trained Keras model, this cannot be modified by the event handler
    - **results** - The model :py:class:`~TrainingResults`
    - **output_dir** - Directory path where the results are saved
    """

    BEFORE_SAVE_TRAIN_ARCHIVE = enum.auto()
    """Invoked during :py:class:`~train_model` before the model archive is saved

    This has the additional keyword arguments:

    - **archive_path** - Path where archive will be saved
    """

    AFTER_SAVE_TRAIN_ARCHIVE = enum.auto()
    """Invoked during :py:class:`~train_model` after the model archive is saved

    This has the additional keyword arguments:

    - **archive_path** - Path where archive was saved
    """

    TRAIN_SHUTDOWN = enum.auto()
    """Invoked at the end of :py:class:`~train_model`

    This has the additional keyword arguments:

    - **results** - The model :py:class:`~TrainingResults`
    """

    QUANTIZE_STARTUP = enum.auto()
    """Invoked at the beginning of :py:class:`~quantize_model`

    This has the additional keyword arguments:

    - **build** - True if the model is being built for profiling
    - **keras_model** - The provided Keras model, if one was given
    - **tflite_converter_settings** - Dictionary of settings that will be given to `TfliteConverter <https://www.tensorflow.org/lite/convert>`_
    - **post_process** - True if post-processing is enabled
    """

    BEFORE_QUANTIZE = enum.auto()
    """Invoked during :py:class:`~quantize_model` before the ` <https://www.tensorflow.org/lite/convert>`_ is invoked

    This has the additional keyword arguments:

    - **converter** - The `TfliteConverter <https://www.tensorflow.org/lite/convert>`_ used to quantize the model
    - **converter_dict** - The `TfliteConverter <https://www.tensorflow.org/lite/convert>`_ as ``converter_dict=dict(value=converter)``, update ``converter_dict['value']`` to return a new converter by the event handler
    """

    AFTER_QUANTIZE = enum.auto()
    """Invoked during :py:class:`~quantize_model` after the TfliteConverter is invoked

    This has the additional keyword arguments:

    - **tflite_flatbuffer** - The tflite flatbuffer binary array
    - **tflite_flatbuffer_dict** - The ``tflite_flatbuffer`` as ``tflite_flatbuffer_dict=dict(value=tflite_flatbuffer)``, update ``tflite_flatbuffer_dict['value']`` to return a new tflite_flatbuffer by the event handler
    - **update_archive** - True if the model archive was updated with the quantized model
    - **keras_model** - The provided Keras model, if one was given
    - **tflite_converter_settings** - Dictionary of settings that will be given to `TfliteConverter <https://www.tensorflow.org/lite/convert>`_
    """

    QUANTIZE_SHUTDOWN = enum.auto()
    """Invoked at the end of :py:class:`~quantize_model`

    This has the additional keyword arguments:

    - **tflite_model** - The quantized :py:class:`~TfliteModel` instance
    - **update_archive** - True if the model archive was updated with the quantized model
    - **keras_model** - The provided Keras model, if one was given
    - **tflite_converter_settings** - Dictionary of settings that will be given to `TfliteConverter <https://www.tensorflow.org/lite/convert>`_
    """

    EVALUATE_STARTUP = enum.auto()
    """Invoked at the beginning of :py:class:`~evaluate_model`

    This has the additional keyword arguments:

    - **tflite** - True if should evaluate `.tflite` model, else evaluating Keras model
    - **max_samples_per_class** - This option places an upper limit on the number of samples per class that are used for evaluation
    - **post_process** - True if post-processing is enabled
    """

    EVALUATE_SHUTDOWN = enum.auto()
    """Invoked at the end of :py:class:`~evaluate_model`

    This has the additional keyword arguments:

    - **results** - The generated :py:class:`~EvaluationResults`
    """

    GENERATE_EVALUATE_PLOT = enum.auto()
    """Invoked when generating a plot during :py:class:`~evaluate_model`

    This has the additional keyword arguments:

    - **tflite** - True if evaluating `.tflite` model, else evaluating Keras model
    - **name** - The name of the plot
    - **fig** - The matlibplot figure
    """

    AFTER_PROFILE = enum.auto()
    """Invoked at the end of :py:class:`~profile_model`

    This has the additional keyword arguments:

    - **results** - The generated :py:class:`~ProfilingModelResults`
    """

    def __str__(self) -> str:
        return self.name


