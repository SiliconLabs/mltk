from typing import List, Callable, Dict, Union, Tuple
import random
import threading
import functools
import os
import time
from multiprocessing.pool import ThreadPool



from mltk.utils.string_formatting import format_units

from .generator_types import (
    BackendBase,
    Voice,
    GenerationConfig,
    Keyword,
    Augmentation,
    logger
)
from .backends import BACKENDS


class AudioDatasetGenerator:
    """Utility for generating synthetic keyword datasets

    See the `Synthetic Audio Dataset Generation <https://siliconlabs.github.io/mltk/mltk/tutorials/synthetic_audio_dataset_generation.html>`_ tutorial for more details.

    .. note:: The generated audio files are 16kHz, 16-bit PCM ``.wav`` files.

    Args:
        out_dir: Directory where dataset will be generated
        n_jobs: Number of parallel processing jobs
    """
    def __init__(self, out_dir:str, n_jobs:int=4):
        self._backends:Dict[str, BackendBase] = {}
        self._out_dir = out_dir
        self._pool = _ProcessingPool(n_jobs=n_jobs)
        self._lock = threading.RLock()
        self._condition = threading.Condition(lock=self._lock)
        self._pending_configs:Dict[BackendBase,List[Tuple[GenerationConfig,Callable]]] = {}
        self._n_config_processing = 0
        t = threading.Thread(
            target=self._generation_loop,
            name='AudioDatasetGenerator',
            daemon=True
        )
        t.start()


    @staticmethod
    def list_supported_backends() -> List[str]:
        """Return a list of the available backends"""
        return list(BACKENDS.keys())

    @property
    def is_running(self) -> bool:
        """Return if the processing pool is active"""
        return self._pool.is_running

    @property
    def out_dir(self) -> bool:
        """Return the output directory where the dataset is generated"""
        return self._out_dir

    def is_backend_loaded(self, backend:str, raise_exception=False) -> bool:
        """Return if the given backend has been loadedd"""
        if backend not in BACKENDS:
            if raise_exception:
                raise ValueError(
                    f'Unknown backend: {backend}, supported backends are: {", ".join(AudioDatasetGenerator.list_supported_backends())}'
                )
            return False

        if backend not in self._backends:
            if raise_exception:
                raise ValueError(f'Backend: {backend} not loaded')
            return False

        return True


    def load_backend(self, name:str, install_python_package=False, **kwargs):
        """Load the specified backend

        NOTE: The backend's corresponding "credentials" must be provided

        Additional kwargs may be passed to the backend's initialization.
        Refer the the backend's docs for the available kwargs:

        - ``name=aws`` --> `boto3.session.Session <https://boto3.amazonaws.com/v1/documentation/api/latest/guide/session.html>`_
        - ``name=azure`` --> `azure.cognitiveservices.speech.SpeechConfig <https://learn.microsoft.com/en-us/python/api/azure-cognitiveservices-speech/azure.cognitiveservices.speech.speechconfig?source=recommendations&view=azure-python>`_
        - ``name=gcp`` --> `google.cloud.texttospeech.TextToSpeechClient <https://cloud.google.com/python/docs/reference/texttospeech/latest/google.cloud.texttospeech_v1beta1.services.text_to_speech.TextToSpeechClient>`_


        Args:
            name: The name of the cloud backend, see :py:func:`~list_supported_backends`
            auto_install_python_package: If true, then automatically install the backend's corresponding Python package (if necessary)
            kwargs: Additional keyword args to pass to the backend's Python package (see comment above)
        """
        if name not in BACKENDS:
            raise ValueError(
                f'Unknown backend: {name}, supported backends are: {", ".join(AudioDatasetGenerator.list_supported_backends())}'
            )

        if self.is_backend_loaded(name):
            raise RuntimeError(f'Backend {name} already loaded')


        backend = BACKENDS[name]()
        backend.load(install_python_package=install_python_package, **kwargs)
        self._backends[name] = backend
        self._pending_configs[backend] = []


    def list_languages(self, backend:str=None) -> List[str]:
        """Return a list of the available language codes

        Args:
            backend: If provided, then only return languages supported by backend,
                else return languages for all loaded backends
        Returns:
            List of languages codes
        """
        retval = set()
        for backend_name in self._get_backend_list(backend):
            for lang in self._backends[backend_name].list_languages():
                retval.add(lang)

        return sorted(retval)


    def list_voices(self, language_code:str=None, backend:str=None) -> List[Voice]:
        """Return a list of the available "voices"

        Args:
            language_code: If provided, then only returned voices that support given language code,
                else return all languages
            backend: If provided, then only return voices supported by backend,
                else return voices for all loaded backends
        Returns:
            List of voices
        """
        retval:List[Voice] = []
        for backend_name in self._get_backend_list(backend):
            retval.extend(self._backends[backend_name].list_voices(language_code=language_code))

        return sorted(retval, key=lambda x: (x.backend, x.language_code, x.name))


    def list_configurations(
        self,
        keywords:List[Keyword],
        augmentations:List[Augmentation],
        voices:List[Voice],
        truncate=False,
        seed:int=None,
    ) -> Dict[Keyword,List[GenerationConfig]]:
        """Generate a list of generation configurations

        Generate a list of all possible combinations of the given keywords, augmentations, and voices.
        If the ``truncate`` argument is provided, then shuffle the generated list and return the truncated list
        based on the ``max_count`` specified in the ``keywords``.

        Args:
            keywords: List of keywords to use for the generation configurations
            augmentations: List of augmentations to apply to each keyword
            voices: List of voices to use for keyword generation
            truncate: If true, then randomly shuffle all possible combinations
                and return a truncated list of configurations. The truncated count is specified in the ``max_count`` field of the keywords
            seed: Seed to use for randomly shuffling the truncated list
        Returns:
            Dictionary of keywords and corresponding list of configurations
        """
        retval:Dict[Keyword,List[GenerationConfig]] = {}

        for keyword in keywords:
            keyword_configs:List[GenerationConfig] = []
            for voice in voices:
                base_configs = self._backends[voice.backend].list_configurations(
                    augmentations=augmentations,
                    voice=voice
                )
                for kw in keyword.as_list():
                    for config in base_configs:
                        config = config.copy()
                        config.keyword = kw
                        config.keyword_group = keyword.value
                        keyword_configs.append(config)

            if truncate and keyword.max_count:
                if seed:
                    random.seed(seed)
                random.shuffle(keyword_configs)
                keyword_configs = keyword_configs[:keyword.max_count]

            retval[keyword] = sorted(keyword_configs, key=lambda x: (x.keyword, x.voice.backend, x.voice.language_code, x.voice.name, x.rate, x.pitch))

        return retval


    def count_characters(
        self,
        config:Dict[Keyword,List[GenerationConfig]],
    ) -> Dict[Keyword,Dict[str,int]]:
        """Count the number of characters that will be sent to each backend

        The cloud backends charge per character that is sent.
        This API returns the number of characters required for each keyword.

        Args:
            config: Dictionary of keywords and corresponding list of configurations
                returned by :py:func:`~list_configurations`
        Returns:
            Dictionary<keyword, Dictionary<backend, char count>>
        """
        retval:Dict[Keyword,Dict[str,int]] = {}

        for keyword, config_list in config.items():
            stats:Dict[str,int] = {}
            for cfg in config_list:
                n_chars = self._backends[cfg.voice.backend].count_characters(cfg)
                stats[cfg.voice.backend] = stats.get(cfg.voice.backend, 0) + n_chars
            retval[keyword] = stats

        return retval


    def get_summary(
        self,
        config:Dict[Keyword,List[GenerationConfig]],
        as_dict=False
    ) -> Union[dict,str]:
        """Generate a summary of the given configurations

        Args:
            config: Dictionary of keywords and corresponding list of configurations
                returned by :py:func:`~list_configurations`
            as_dict: If true then return the summary as a dictionary,
                else return the summary as a string
        Returns:
            If ``as_dict=True`` then return the summary as a dictionary,
                else return the summary as a string
        """
        voice_stats = {}
        sample_stats = {}

        voices = set()
        for config_list in config.values():
            for cfg in config_list:
                if cfg.voice not in voices:
                    backend = cfg.voice.backend
                    voices.add(cfg.voice)
                    n_voices = voice_stats.get(backend, 0) + 1
                    voice_stats[backend] = n_voices

        for keyword, config_list in config.items():
            stats = {}
            for cfg in config_list:
                backend = cfg.voice.backend
                n_samples = stats.get(backend, 0) + 1
                stats[backend] = n_samples

            sample_stats[keyword] = stats

        character_stats = self.count_characters(config)

        retval = dict(
            voices=voice_stats,
            samples=sample_stats,
            characters=character_stats
        )
        if as_dict:
            return retval


        s  = 'Voice Counts\n'
        s += '---------------------\n'
        n_voices = 0
        for backend, count in retval['voices'].items():
            s += f'  {backend:<6s}: {count}\n'
            n_voices += count
        s += f'  Total : {n_voices}\n'

        s += '\n'
        s += 'Keyword Counts\n'
        s += '---------------------\n'
        total_samples = 0
        for keyword, stats in retval['samples'].items():
            n_samples = 0
            s += f'  {keyword}:\n'
            for backend, count in stats.items():
                s += f'    {backend:<6s}: {format_units(count, add_space=False, precision=1)}\n'
                n_samples += count
                total_samples += count
            s += f'    Total : {format_units(n_samples, add_space=False, precision=1)}\n'
        s += f'  Overall total: {format_units(total_samples, add_space=False, precision=1)}\n'

        s += '\n'
        s += 'Character Counts\n'
        s += '---------------------\n'
        backend_totals = {}
        for keyword, stats in retval['characters'].items():
            s += f'  {keyword}:\n'
            for backend, count in stats.items():
                s += f'    {backend:<6s}: {format_units(count, add_space=False, precision=1)}\n'
                backend_totals[backend] = backend_totals.get(backend, 0) + count

        s += '  Backend totals:\n'
        for backend, count in backend_totals.items():
            s += f'    {backend:<6s}: {format_units(count, add_space=False, precision=1)}\n'

        return s


    def generate(
        self,
        config:GenerationConfig,
        on_finished:Callable[[str],None]=None
    ):
        """Generate a keyword using the given configuration

        This will generate a keyword using the given configuration
        in the specified :py:attr:`~out_dir`.
        Processing is done asynchronously in a thread pool.
        The ``on_finished`` will be invoked when processing is complete.
        Alternatively, call :py:func:`~join` to wait for all processing to complete.

        Args:
            config: The configuration to use for keyword generation
            on_finished: Optional callback to be invoked when generation completes
                The parameter given to the callback contains the file path to the generated audio file
        """
        if not self.is_running:
            raise RuntimeError('Not running')

        with self._lock:
            backend = self._backends[config.voice.backend]
            self._pending_configs[backend].append((config, on_finished))
            self._condition.notify_all()



    def join(self, timeout:float=None) -> bool:
        """Wait for all generation tasks to complete

        Args:
            timeout: The maximum amount of time in seconds to wait
                If not specified then wait forever

        Returns:
            True if processing has completed, false else
        """
        is_processing = True
        start_time = time.time()
        with self._lock:
            while is_processing and self.is_running:
                is_processing = False
                for configs in self._pending_configs.values():
                    if self._n_config_processing > 0 or len(configs) > 0:
                        is_processing = True
                        break

                if not is_processing or (timeout and (time.time() - start_time) > timeout):
                    break

                self._condition.wait(0.100)

        return not is_processing


    def shutdown(self):
        """Shutdown the underlying thread pool"""
        self._pool.shutdown()


    def _get_backend_list(self, backend:str=None) -> List[str]:
        if backend:
            self.is_backend_loaded(backend, raise_exception=True)

        if len(self._backends) == 0:
            raise RuntimeError('No backends loaded')

        return [backend] if backend is not None else list(self._backends.keys())



    def _generation_loop(self):
        while self.is_running:
            with self._lock:
                try:
                    self._condition.wait(timeout=0.010)
                except:
                    pass
                self._process_once()


    def _process_once(self):
        if not self.is_running:
            return

        for backend, configs in self._pending_configs.items():
            if len(configs) == 0 or backend.is_rate_limited:
                continue
            self._n_config_processing += 1
            config, callback = configs.pop(0)
            out_dir = f'{self._out_dir}/{config.keyword_group}'
            os.makedirs(out_dir, exist_ok=True)
            self._pool(
                backend.generate,
                config=config,
                out_dir=out_dir,
                _on_finished=functools.partial(self._on_finished, callback=callback),
                _on_error=self._on_error
            )

    def _on_finished(self, sample_path:str, callback:Callable=None):
        with self._lock:
            self._n_config_processing -= 1
            self._condition.notify_all()
        if callback is not None:
            try:
                callback(sample_path)
            except:
                pass

    def _on_error(self, e:Exception):
        with self._lock:
            logger.exception(f'{e}', exc_info=e)
            self._n_config_processing -= 1
            self._condition.notify_all()

    def __enter__(self):
        return self

    def __exit__(self, dtype, value, tb):
        self.join()
        self.shutdown()



class _ProcessingPool():

    def __init__(self, n_jobs:int):
        self.pool = ThreadPool(processes=n_jobs)
        self._running_event = threading.Event()

    @property
    def is_running(self) -> bool:
        return not self._running_event.is_set()


    def shutdown(self):
        self._running_event.set()
        self.pool.close()

    def __call__(self, func, *, _on_finished, _on_error, **kwargs):
        self.pool.apply_async(func, kwds=kwargs, callback=_on_finished, error_callback=_on_error)


