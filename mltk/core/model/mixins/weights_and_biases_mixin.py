from typing import Optional, Dict, Any
import os
import shutil
import copy
import logging
import warnings



try:
    import wandb
    from wandb.keras import WandbCallback
    from wandb.keras import WandbModelCheckpoint
except:
    class WandbCallback:
        '''Placeholder for https://docs.wandb.ai/ref/python/integrations/keras/wandbcallback'''
    class WandbModelCheckpoint:
        '''Placeholder for https://docs.wandb.ai/ref/python/integrations/keras/wandbmodelcheckpoint'''


from mltk.utils.python import prepend_exception_msg
from mltk.utils.string_formatting import iso_time_filename_str
from mltk.core.utils import get_mltk_logger
from mltk.core.profiling_results import ProfilingModelResults
from mltk.core.evaluation_results import EvaluationResults

from .base_mixin import BaseMixin
from ..model_attributes import MltkModelAttributesDecorator
from ..model_event import MltkModelEvent


@MltkModelAttributesDecorator()
class WeightsAndBiasesMixin(BaseMixin):
    """Provides various properties to the base :py:class:`~MltkModel` used by `Weights & Biases <https://wandb.ai>`_
    3rd-party cloud backend.

    .. seealso::

       - `Tutorial: Cloud logging with Weights & Biases <https://siliconlabs.github.io/mltk/mltk/tutorials/cloud_logging_with_wandb.html>`_
       - `Weights & Biases Documentation <https://docs.wandb.ai>`_
    
    """

    @property
    def wandb_is_initialized(self) -> bool:
        """Return if the wandb backend is initialized"""
        return self._attributes.get_value('wandb.is_initialized', default=False)


    @property
    def wandb_is_disabled(self) -> bool:
        """Manually disable the wandb backend"""
        return self._attributes.get_value('wandb.is_disabled', default=False)
    @wandb_is_disabled.setter
    def wandb_is_disabled(self, v:bool):
        self._attributes['wandb.is_disabled'] = v


    @property
    def wandb_init_kwargs(self) -> dict:
        """Additional arguments to provide to ``wandb.init()``

        The following argument are automatically populated by this mixin:

        - **project** - The name of the :py:attr:`~MltkModel.name`
        - **job_type** - ``train``, ``evaluation``, ``quantize``, or ``profile``
        - **dir** - The :py:attr:`~MltkModel.log_dir`
        - **id** - The timestamp when the training was invoked. See :py:attr:`~WeightsAndBiasesMixin.wandb_session_id`  This id is re-used by the ``evaluate``, ``profile``, and ``quantize`` commands
        - **resume** - Set to ``never`` for the ``train`` command, and ``must`` otherwise

        See `wandb.init() <https://docs.wandb.ai/ref/python/init>`_ for the other available arguments
        """
        return self._attributes.get_value('wandb.init_kwargs', default={})
    @wandb_init_kwargs.setter
    def wandb_init_kwargs(self, v:dict):
        self._attributes['wandb.init_kwargs'] = v


    @property
    def wandb_config(self) -> dict:
        """Additional configuration values

        This sets the `wandb.config <https://docs.wandb.ai/guides/track/config>`_ object in your script to save your training configuration:
        hyperparameters, input settings like dataset name or model type, and any other independent variables for your experiments.
        This is useful for analyzing your experiments and reproducing your work in the future.
        You'll be able to group by config values in the web interface, comparing the settings of different runs and seeing how these affect the output.
        """
        return self._attributes.get_value('wandb.config', default={})
    @wandb_config.setter
    def wandb_config(self, v:dict):
        self._attributes['wandb.config'] = v


    @property
    def wandb_callback(self) -> WandbCallback:
        """Keras callback to automatically log info to wandb

        This allows for specifying a custom `wandb.keras.WandbCallback <https://docs.wandb.ai/ref/python/integrations/keras/wandbcallback>`_.
        If not set, then the mixin will automatically populate this callback.
        """
        return self._attributes.get_value('wandb.callback', default=None)
    @wandb_callback.setter
    def wandb_callback(self, v:WandbCallback):
        self._attributes['wandb.callback'] = v

    @property
    def wandb_model_checkpoint_callback(self) -> WandbModelCheckpoint:
        """Callback to periodically save the Keras model or model weights

        This allows for specifying a custom `wandb.keras.WandbModelCheckpoint <https://docs.wandb.ai/ref/python/integrations/keras/wandbmodelcheckpoint>`_.
        """
        return self._attributes.get_value('wandb.model_checkpoint_callback', default=None)
    @wandb_model_checkpoint_callback.setter
    def wandb_model_checkpoint_callback(self, v:WandbModelCheckpoint):
        self._attributes['wandb.model_checkpoint_callback'] = v

    @property
    def wandb_session_id(self) -> str:
        """The wandb project session or run ID

        This is the timestamp of when the last `train` command was invoked for the model.
        This ID is re-used for `evaluate`, `profile`, and `quantize` commands.

        This value is used at the ``id`` argument to `wandb.init() <https://docs.wandb.ai/ref/python/init>`_
        """
        return self._attributes.get_value('wandb.session_id', default=None)


    def wandb_save(
        self,
        glob_str: Optional[str] = None,
        base_path: Optional[str] = None,
        policy = "live",
        logger:logging.Logger = None
    ):
        """Save files to wandb cloud

        Internally, this invokes `wandb.save() <https://docs.wandb.ai/ref/python/save>`_

        Args:
            glob_str: a relative or absolute path to a unix glob or regular path. If this isn't specified the method is a noop.
            base_path: the base path to run the glob relative to
            policy: one of ``live``, ``now``, or ``end``

                - ``live``: upload the file as it changes, overwriting the previous version
                - ``now``: upload the file once now
                - ``end``: only upload file when the run ends
        """
        if not self.wandb_is_initialized:
            return

        logger = logger or get_mltk_logger()

        try:
            logger.debug(f'Saving to wandb: glob_str={glob_str}, base_path={base_path}')
            wandb.save(
                glob_str=glob_str,
                base_path=base_path,
                policy=policy
            )
        except Exception as e:
           get_mltk_logger() .error(f'Failed to save to wandb cloud, glob_str={glob_str}, base_path={base_path}, err: {e}')


    def wandb_log(
        self,
        data: Dict[str, Any],
        step: Optional[int] = None,
        commit: Optional[bool] = True,
        logger:logging.Logger = None
    ):
        """Logs a dictionary of data to the current wandb run's history

        Internally, this invokes `wandb.log() <https://docs.wandb.ai/ref/python/log>`_

        Args:
            data: A dict of serializable python objects i.e str, ints, floats, Tensors, dicts, or any of the wandb.data_types
            commit: Save the metrics dict to the wandb server and increment the step.
                If false wandb.log just updates the current metrics dict with the data argument and metrics won't be saved until wandb.log is called with commit=True.
            step: The global step in processing. This persists any non-committed earlier steps but defaults to not committing the specified step.
        """
        assert isinstance(data, dict), 'Data not instance of dict'
        if not self.wandb_is_initialized:
            return
        logger = logger or get_mltk_logger()
        logger.debug(f'Logging to wandb: data keys={", ".join(data)}')
        try:
            wandb.log(
                data=data,
                step=step,
                commit=commit,
            )
        except Exception as e:
            logger.error(f'Failed to log to wandb cloud, err: {e}')


    def _register_attributes(self):
        """This is called when the MltkModel properties are first registered,
        see _check_attributes_registered_decorator()
        """
        try:
            import wandb
        except:
            get_mltk_logger().warning('Failed import wandb Python package, try running: pip install wandb plotly')
            return

        get_mltk_logger().debug('Registering WeightsAndBiasesMixin')

        # Register the various properties for this mixin
        self._attributes.register('wandb.init_kwargs', dtype=dict)
        self._attributes.register('wandb.callback', dtype=WandbCallback)
        self._attributes.register('wandb.model_checkpoint_callback', dtype=WandbModelCheckpoint)
        self._attributes.register('wandb.config', dtype=dict)
        self._attributes.register('wandb.session_id', dtype=str)
        self._attributes.register('wandb.is_initialized', dtype=bool)
        self._attributes.register('wandb.is_disabled', dtype=bool)

        # Register the various model event handlers
        self.add_event_handler(
            MltkModelEvent.AFTER_MODEL_LOAD,
            self._wandb_load
        )

        self.add_event_handler(
            MltkModelEvent.TRAIN_STARTUP,
            self._wandb_init,
            job_type='train'
        )

        self.add_event_handler(
            MltkModelEvent.SUMMARIZE_MODEL,
            self._wandb_log_model_summary,
        )

        self.add_event_handler(
            MltkModelEvent.SUMMARIZE_DATASET,
            self._wandb_log_dataset_summary,
        )

        self.add_event_handler(
            MltkModelEvent.EVALUATE_STARTUP,
            self._wandb_init,
            job_type='evaluate'
        )

        self.add_event_handler(
            MltkModelEvent.QUANTIZE_STARTUP,
            self._wandb_init,
            job_type='quantize'
        )

        self.add_event_handler(
            MltkModelEvent.POPULATE_TRAIN_CALLBACKS,
            self._wandb_populate_train_callbacks
        )

        self.add_event_handler(
            MltkModelEvent.QUANTIZE_SHUTDOWN,
            self._wandb_save_archive
        )

        self.add_event_handler(
            MltkModelEvent.GENERATE_EVALUATE_PLOT,
            self._wandb_save_plot
        )

        self.add_event_handler(
            MltkModelEvent.EVALUATE_SHUTDOWN,
            self._wandb_log_evaluation_summary
        )

        self.add_event_handler(
            MltkModelEvent.AFTER_PROFILE,
            self._wandb_upload_profiling_results
        )

    def _wandb_load(self, logger:logging.Logger, **kwargs):
        """This is called after the MltkModel is loaded"""
        try:
            wandb_session_id_path = self.get_archive_file('wandb/session_id.txt')
            with open(wandb_session_id_path, 'r') as f:
                session_id = f.read().strip()
                self._attributes['wandb.session_id'] = session_id
                logger.debug(f'wandb session id: {self.wandb_session_id}')
        except:
            pass


    def _wandb_init(self, job_type:str, post_process:bool, logger:logging.Logger, **kwargs):
        """This is called at the beginning of train_model(), evaluate_model(), or quantize_model()"""
        import absl.logging

        # Do not initialize if:
        # - User didn't add the --post arg to the command
        # - Or we're building a temp model
        if not post_process or (job_type == 'quantize' and kwargs.get('build', False)):
            if not post_process:
                logger.debug('No post_processing enabled (e.g. train my_model --post), so not initializing wandb')
            self._attributes['wandb.is_disabled'] = True
            return

        if self.wandb_is_initialized or self.wandb_is_disabled:
            return

        if job_type == 'train':
            resume = 'never'
            self._attributes['wandb.session_id'] = iso_time_filename_str()
            os.makedirs(f'{self.log_dir}/wandb', exist_ok=True)
            with open(f'{self.log_dir}/wandb/session_id.txt', 'w') as f:
                f.write(self.wandb_session_id)
                logger.debug(f'wandb session id: {self.wandb_session_id}')
        else:
            resume = 'must'

        if self.wandb_session_id is None:
            return

        try:
            logger.info('Initializing wandb')
            init_kwargs = self.wandb_init_kwargs
            settings = init_kwargs.pop('settings', dict(show_info=False))
            wandb.init(
                project=self.name if not self.test_mode_enabled else f'{self.name}-test',
                job_type=job_type,
                dir=self.log_dir,
                id=self.wandb_session_id,
                resume=resume,
                settings=settings,
                **init_kwargs
            )
        except Exception as e:
            prepend_exception_msg(e, 'Failed to init wandb')
            raise

        if job_type == 'train':
            absl.logging.set_verbosity('ERROR')

            config = self.wandb_config
            if config is not None:
                config['epochs'] = self.epochs
                config['batch_size'] = self.batch_size
                config['classes'] = self.classes
                config['class_weights'] = self.class_weights
                wandb.config.update(config)

        self._attributes['wandb.is_initialized'] = True
        logger.info('wandb initialized')


    def _wandb_populate_train_callbacks(self, keras_callbacks:list, **kwargs):
        """This is called after train_model() populates the various Keras callbacks, but before training starts"""
        if not self.wandb_is_initialized:
            return

        callback = WandbCallback() if self.wandb_callback is None else self.wandb_callback
        if callback:
            callback = WandbCallback()
            keras_callbacks.append(callback)

        weights_dir =  f'{self.log_dir}/train/weights'
        os.makedirs(weights_dir, exist_ok=True)
        weights_file_format = self.weights_file_format
        weights_path = f'{weights_dir}/{weights_file_format}'

        checkpoint_callback = self.wandb_model_checkpoint_callback
        if checkpoint_callback is not None:
            checkpoint_callback.filepath = weights_path
            keras_callbacks.append(checkpoint_callback)


    def _wandb_log_model_summary(self, summary:str, logger:logging.Logger, **kwargs):
        """Log the model summary to wandb if we're training"""
        if self.loaded_subset == 'training':
            self.wandb_log({'model_summary': _generate_html(summary)}, logger=logger)


    def _wandb_log_dataset_summary(self, summary:str, logger:logging.Logger, **kwargs):
        """Log the dataset summary to wandb"""
        self.wandb_log({f'dataset_summary-{self.loaded_subset}': _generate_html(summary)}, logger=logger)


    def _wandb_log_evaluation_summary(
        self,
        results:EvaluationResults,
        tflite:bool,
        logger:logging.Logger,
        **kwargs
    ):
        """Log the evaluation summary to wandb"""
        summary = results.generate_summary()
        eval_type = 'tflite' if tflite else 'keras'
        self.wandb_log({f'eval_summary-{eval_type}': _generate_html(summary)}, logger=logger)
        self._wandb_save_archive(logger=logger)


    def _wandb_save_archive(self, logger:logging.Logger, **kwargs):
        """Upload the model archive to wandb"""
        if not self.wandb_is_initialized:
            return

        self.add_archive_file(f'{self.log_dir}/wandb/session_id.txt')
        archive_path = self.archive_path
        model_specification_path = self.model_specification_path
        dst_dir = wandb.run.settings.files_dir
        shutil.copy(archive_path, dst_dir)
        shutil.copy(model_specification_path, dst_dir)
        logger.debug(f'Uploading to wandb: {archive_path}')
        self.wandb_save(f'{dst_dir}/{os.path.basename(archive_path)}', logger=logger)
        self.wandb_save(f'{dst_dir}/{os.path.basename(model_specification_path)}', logger=logger)


    def _wandb_save_plot(self, name, fig, tflite:bool, logger:logging.Logger, **kwargs):
        """Log an evaluation plot to wandb"""
        if not self.wandb_is_initialized:
            return

        name += ('-tflite' if tflite else '-keras')
        logger.debug(f'Saving wandb plot: {name}')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.wandb_log( {name : copy.deepcopy(fig)})


    def _wandb_upload_profiling_results(
        self,
        results:ProfilingModelResults,
        logger:logging.Logger,
        **kwargs
    ):
        """Log the profiling results to wandb"""
        self._wandb_init(
            job_type='profile',
            logger=logger,
            post_process=True
        )
        if not self.wandb_is_initialized:
            return

        accelerator = results.accelerator or 'cmsis'

        self.wandb_log({f'profiling_report-{accelerator}': _generate_html(results.to_string())})

        results_dict = results.to_dict()
        layers = results_dict['layers']
        layers_headers = list(layers[0].keys())[1:]
        layers_data = []
        for layer in layers:
            row = []
            for key in layers_headers:
                row.append(layer[key])
            layers_data.append(row)

        layer_table = wandb.Table(data=layers_data, columns=layers_headers, allow_mixed_types=True)
        self.wandb_log({f'profiling_layers_table-{accelerator}': layer_table})


def _generate_html(data:str):
    return wandb.Html(
        f'<pre class="background-white">{data}</pre>'
    )