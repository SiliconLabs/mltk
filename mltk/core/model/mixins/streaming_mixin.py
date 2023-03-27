from __future__ import annotations
from typing import Tuple, List
import os
import logging
import shutil
import warnings
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from .base_mixin import BaseMixin


from ..model_event import MltkModelEvent
from ..model_attributes import MltkModelAttributesDecorator


@MltkModelAttributesDecorator()
class StreamingMixin(BaseMixin):
    """
    """

    @property
    def streaming_input_shape(self) -> Tuple[int]:
        input_shape = self.input_shape # <n_spectrogram_slices, n_freq_bins, 1>
        return (1, *input_shape[1:])

    @property
    def non_streaming_tflite_log_dir_path(self) -> str:
        tflite_path = self.tflite_log_dir_path
        return tflite_path[:-len('.tflite')] + '.non_streaming.tflite'

    def _register_attributes(self):
        try:
            import tensorflow_model_optimization # pylint:disable=unused-import
        except ImportError:
            raise RuntimeError(
                '\n\n\n*** You must first install the tensorflow_model_optimization package to use StreamingMixin\n'
                'pip install tensorflow_model_optimization\n\n'
            )

        self.add_event_handler(
            MltkModelEvent.AFTER_QUANTIZE,
            self._create_streaming_model,
            _raise_exception=True
        )
        self.add_event_handler(
            MltkModelEvent.AFTER_MODEL_LOAD,
            self._initialize_streaming_properties
        )


    def _initialize_streaming_properties(self, **kwargs):
        from mltk.core import EvaluateClassifierMixin
        from mltk.models.shared.kws_streaming.layers import stream
        self.keras_custom_objects['Stream'] = stream.Stream

        if isinstance(self, EvaluateClassifierMixin):
            # Use the custom streaming classifier function
            self.eval_custom_function = evaluate_streaming_classifier


    def _create_streaming_model(
        self,
        keras_model:tf.keras.Model,
        tflite_converter_settings:dict,
        update_archive:bool,
        logger:logging.Logger,
        tflite_flatbuffer_dict:dict,
        **kwargs
    ):
        from mltk.models.shared.kws_streaming.models.utils import convert_to_inference_model
        from mltk.core import summarize_model, load_tflite_model, TfliteModel

        from mltk.core.quantize_model import (
            populate_converter_options,
        )

        output_tflite_path = self.tflite_log_dir_path
        non_streaming_tflite_path = self.non_streaming_tflite_log_dir_path
        logger.info(f'Saving {output_tflite_path} to {non_streaming_tflite_path}')
        with open(non_streaming_tflite_path, 'wb') as f:
            f.write(tflite_flatbuffer_dict['value'])

        if update_archive:
            self.add_archive_file(non_streaming_tflite_path)

            try:
                summary_path = f'{non_streaming_tflite_path}.summary.txt'
                with open(summary_path, 'w') as fp:
                    tflite_model = load_tflite_model(non_streaming_tflite_path)
                    fp.write(summarize_model(tflite_model))
                self.add_archive_file(summary_path)
            except:
                pass

        logger.info(f'Generating streaming: {output_tflite_path} ...')

        if isinstance(keras_model.input, (tuple, list)):
            dtype = keras_model.input[0].dtype
        else:
            dtype = keras_model.input.dtype


        try:
            self.load_dataset(subset='validation', test=self.test_mode_enabled, logger=logger)
            representative_data = get_representative_data(
                self,
                max_samples=3 if self.test_mode_enabled else 100,
                logger=logger
            )
        finally:
            self.unload_dataset()

        tf_model_path = self.create_log_dir('tf_model', delete_existing=True)
        logger.info(f'Saving non-streaming TF model to {tf_model_path}')
        tf.keras.models.save_model(keras_model, tf_model_path)


        logger.info('Generating streaming model ...')
        with tf.compat.v1.Session() as sess, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            input_shape = self.streaming_input_shape
            input_tensors = [
                tf.keras.layers.Input(
                    shape=input_shape,
                    dtype=dtype,
                    name='input_data',
                    batch_size=1
                )
            ]

            loaded_model = tf.keras.models.load_model(tf_model_path, custom_objects=self.keras_custom_objects)
            inference_model = convert_to_inference_model(
                loaded_model,
                input_tensors,
                mode='STREAM_EXTERNAL_STATE_INFERENCE'
            )
            inference_model.train = False


            converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, inference_model.inputs, inference_model.outputs)

            populate_converter_options(converter, tflite_converter_settings=tflite_converter_settings)
            converter.experimental_new_converter = True
            converter.experimental_new_quantizer = True
            converter.experimental_enable_resource_variables = True
            converter.representative_dataset = get_stateful_representative_datagen(
                self,
                representative_data=representative_data,
                inference_model=inference_model,
                logger=logger
            )

            try:
                logger.info('Generating .tflite from streaming model ...')
                tflite_flatbuffer = converter.convert()
            finally:
                # The TfLiteConverter adds a StreamHandler to the root logger,
                # remove it so we don't double print everything to the console
                logger.root.handlers.clear()


        tflite_model = TfliteModel(tflite_flatbuffer)
        #optimize_streaming_tflite(tflite_model)
        tflite_flatbuffer_dict['value'] = tflite_model.flatbuffer_data



def evaluate_streaming_classifier(
    mltk_model,
    built_model,
    eval_dir:str,
    logger:logging.Logger,
    show:bool
):
    """Custom callback to evaluate the trained model
    """
    from mltk.core import MltkModel
    from mltk.core.tflite_model import TfliteModel
    from mltk.core.evaluate_classifier import (
        ClassifierEvaluationResults,
        get_progbar,
        iterate_evaluation_data,
        list_to_numpy_array,
        evaluate_classifier_with_built_model
    )

    mltk_model:MltkModel = mltk_model
    if isinstance(built_model, tf.keras.Model):
        return evaluate_classifier_with_built_model(
            mltk_model=mltk_model,
            built_model=built_model,
            logger=logger,
            verbose=True,
            show=show,
        )

    built_model:TfliteModel = built_model
    non_streaming_tflite_path = mltk_model.non_streaming_tflite_log_dir_path
    non_stream_tflite_path = mltk_model.get_archive_file(os.path.basename(non_streaming_tflite_path))
    non_stream_tflite = TfliteModel.load_flatbuffer_file(non_stream_tflite_path)

    stream_y_pred = []
    non_stream_y_pred = []
    y_label = []

    with get_progbar(mltk_model, verbose=True) as progbar:
        for batch_x, batch_y in iterate_evaluation_data(mltk_model):
            if batch_y.shape[-1] == 1 or len(batch_y.shape) == 1:
                y_label.extend(batch_y)
            else:
                y_label.extend(np.argmax(batch_y, -1))

            non_stream_pred = non_stream_tflite.predict(batch_x, y_dtype=np.float32)
            non_stream_y_pred.extend(non_stream_pred)

            interpreter_kwargs = dict(
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF
            )

            for x in batch_x:
                init_state = []
                for inp in built_model.inputs[1:-1]:
                    init_state.append(np.zeros(inp.shape[1:], dtype=inp.dtype))

                for i, row_x in enumerate(x):
                    # if i == 0:
                    #     d = [np.expand_dims(row_x, axis=0), *init_state]
                    # else:
                    #     d = {0: np.expand_dims(row_x, axis=0)}
                    d = [np.expand_dims(row_x, axis=0), *init_state]
                    res = built_model.predict(d, interpreter_kwargs=interpreter_kwargs)
                    init_state = res[1:-1]

                # Convert the data type to float32
                y_pred = res[0]
                y_pred = built_model.dequantize_output_to_float32(y_pred, output_index=0)
                stream_y_pred.append(y_pred)

            progbar.update(len(non_stream_pred))


    y_label = np.asarray(y_label, dtype=np.int32)
    non_stream_y_pred = list_to_numpy_array(non_stream_y_pred)
    stream_y_pred = list_to_numpy_array(stream_y_pred)

    non_streaming_results = ClassifierEvaluationResults(
        name=mltk_model.name + '-non_streaming',
        classes=mltk_model.classes
    )
    streaming_results = ClassifierEvaluationResults(
        name=mltk_model.name,
        classes=mltk_model.classes
    )

    non_streaming_results.calculate(
        y=y_label,
        y_pred=non_stream_y_pred,
    )

    streaming_results.calculate(
        y=y_label,
        y_pred=stream_y_pred,
    )

    non_streaming_eval_dir = f'{eval_dir}/non_streaming'
    os.makedirs(non_streaming_eval_dir, exist_ok=True)
    summary_path = f'{non_streaming_eval_dir}/summary.txt'
    with open(summary_path, 'w') as f:
        f.write(non_streaming_results.generate_summary())
    logger.debug(f'Generated {summary_path}')

    non_streaming_results.generate_plots(
        logger=logger,
        output_dir=non_streaming_eval_dir,
        show=show
    )

    streaming_results.generate_plots(
        logger=logger,
        output_dir=non_streaming_eval_dir,
        show=show
    )

    return streaming_results



def optimize_streaming_tflite(tflite_model):
    from mltk.core.tflite_model import TfliteModel

    tflite_model:TfliteModel = tflite_model
    tensor_mapping = OrderedDict()
    model = tflite_model.flatbuffer_model
    subgraph = model.subgraphs[0]

    tensor_mapping = OrderedDict()

    for inp, outp in zip(subgraph.inputs[1:], subgraph.outputs[1:]):
        has_link = False
        for op in subgraph.operators:
            if inp in op.inputs or inp in op.outputs:
                has_link = True
                break

        if has_link:
            tensor_mapping[outp] = inp

    subgraph.inputs = subgraph.inputs[:-1]
    subgraph.outputs = [subgraph.outputs[0], *subgraph.inputs[1:]]

    for op in subgraph.operators:
        outputs = []
        for o in op.outputs:
            if o in tensor_mapping:
                outputs.append(tensor_mapping[o])
            else:
                outputs.append(o)

        op.outputs = outputs

        inputs = []
        for i in op.inputs:
            if i in tensor_mapping:
                inputs.append(tensor_mapping[i])
            else:
                inputs.append(i)

        op.inputs = inputs


    tflite_model.regenerate_flatbuffer()


def get_representative_data(
    mltk_model,
    logger:logging.Logger,
    max_samples=100,
):
    logger.info('Generating representative dataset using validation dataset')

    # Try to use the validation data if available, otherwise use the training data
    validation_data = mltk_model.validation_data
    if validation_data is None:
        validation_data = mltk_model.x

    retval = []

    for i, batch in enumerate(validation_data):
        batch_x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(batch)
        if isinstance(batch_x, tf.Tensor):
            batch_x = batch_x.numpy()
            if batch_x.dtype != np.float32:
                batch_x = batch_x.astype(np.float32)

            # The TF-Lite converter expects 1 sample batches
            for x in batch_x:
                retval.append(x)
                if len(retval) >= max_samples:
                    return retval
        else:
            # The TF-Lite converter expects the input to be a float32 data type
            if isinstance(batch_x, np.ndarray) and batch_x.dtype != np.float32:
                batch_x = batch_x.astype(np.float32)

            elif isinstance(batch_x, (list,tuple)):
                batch_x_norm = []
                for batch_xi in batch_x:
                    if isinstance(batch_xi, np.ndarray) and batch_xi.dtype != np.float32:
                        batch_x_norm.append(batch_xi.astype(np.float32))
                    else:
                        batch_x_norm.append(batch_xi)
                batch_x = batch_x_norm

            # The TF-Lite converter expects 1 sample batches
            for x in batch_x:
                retval.append(x)
                if len(retval) >= max_samples:
                    return retval

    return retval


def get_stateful_representative_datagen(
    mltk_model,
    representative_data:List[np.ndarray],
    inference_model: tf.keras.Model,
    logger:logging.Logger,
):
    def _data_gen():
        logger.warning('Generating stateful representative dataset (this may take awhile ...)')
        prev_state = []
        for inp in inference_model.inputs[1:]:
            prev_state.append(np.zeros(inp.shape, dtype=np.float32))

        for x in representative_data:
            for row_x in x:
                row_x = np.expand_dims(np.expand_dims(row_x, axis=0), axis=0)
                d = [row_x, *prev_state]
                yield d
                y = inference_model.predict(d)
                prev_state = y[1:]

    return _data_gen

