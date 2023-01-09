import importlib
import numpy as np

from .audio_feature_generator_settings import AudioFeatureGeneratorSettings


class AudioFeatureGenerator:
    """Converts raw audio into a spectrogram (gray-scale 2D image)

    **Example Usage**

    .. highlight:: python
    .. code-block:: python

        import numpy as np
        from mltk.core.preprocess.audio.audio_feature_generator import AudioFeatureGeneratorSettings
        from mltk.core.preprocess.utils import audio as audio_utils

        # Define the settings used to convert the audio into a spectrogram
        frontend_settings = AudioFeatureGeneratorSettings()

        frontend_settings.sample_rate_hz = 16000
        frontend_settings.sample_length_ms = 1200
        frontend_settings.window_size_ms = 30
        frontend_settings.window_step_ms = 10
        frontend_settings.filterbank_n_channels = 108
        frontend_settings.filterbank_upper_band_limit = 7500.0
        frontend_settings.filterbank_lower_band_limit = 125.0
        frontend_settings.noise_reduction_enable = True
        frontend_settings.noise_reduction_smoothing_bits = 10
        frontend_settings.noise_reduction_even_smoothing =  0.025
        frontend_settings.noise_reduction_odd_smoothing = 0.06
        frontend_settings.noise_reduction_min_signal_remaining = 0.40
        frontend_settings.quantize_dynamic_scale_enable = True # Enable dynamic quantization
        frontend_settings.quantize_dynamic_scale_range_db = 40.0

        # Read the raw audio file
        sample, original_sample_rate = audio_utils.read_audio_file(
            'my_audio.wav',
            return_numpy=True,
            return_sample_rate=True
        )

        # Clip/pad the audio so that it's length matches the values configured in "frontend_settings"
        out_length = int((original_sample_rate * frontend_settings.sample_length_ms) / 1000)
        sample = audio_utils.adjust_length(
            sample,
            out_length=out_length,
            trim_threshold_db=30,
            offset=np.random.uniform(0, 1)
        )

        # Convert the sample rate (if necessary)
        if original_sample_rate != frontend_settings.sample_rate_hz:
            sample = audio_utils.resample(
                sample,
                orig_sr=original_sample_rate,
                target_sr=frontend_settings.sample_rate_hz
            )

        # Generate a spectrogram from the audio sample
        #
        # NOTE: audio_utils.apply_frontend() is a helper function.
        #       Internally, it converts from float32 to int16 (audio_utils.read_audio_file() returns float32)
        #       then calls the AudioFeatureGenerator, e.g.:
        #       sample = sample * 32768
        #       sample = sample.astype(np.int16)
        #       sample = np.squeeze(sample, axis=-1)
        #       frontend = AudioFeatureGenerator(frontend_settings)
        #       spectrogram = frontend.process_sample(sample, dtype=np.int8)

        spectrogram = audio_utils.apply_frontend(
            sample=sample,
            settings=frontend_settings,
            dtype=np.int8
        )

    .. seealso::
       - `AudioFeatureGenerator documentation <https://siliconlabs.github.io/mltk/docs/audio/audio_feature_generator.html>`_
       - `AudioFeatureGenerator Python Wrapper <https://siliconlabs.github.io/mltk/docs/cpp_development/wrappers/audio_feature_generator_wrapper.html>`_
       - `Microfrontend implementation <https://github.com/siliconlabs/mltk/tree/master/cpp/shared/microfrontend>`_
       - `ParallelAudioDataGenerator API docs <https://siliconlabs.github.io/mltk/docs/python_api/data_preprocessing/audio_data_generator.html>`_
    """

    def __init__(self, settings: AudioFeatureGeneratorSettings):
        """
        Args:
            settings: The settings to use for processing the audio sample
        """
        try:
            wrapper_module = importlib.import_module('mltk.core.preprocess.audio.audio_feature_generator._audio_feature_generator_wrapper')
        except (ImportError, ModuleNotFoundError) as e:
            raise ImportError(f'Failed to import the AudioFeatureGenerator wrapper C++ shared library, err: {e}\n' \
                            'This likely means you need to re-build the AudioFeatureGenerator wrapper package\n\n') from e

        self._spectrogram_shape = settings.spectrogram_shape
        self._wrapper = wrapper_module.AudioFeatureGeneratorWrapper(settings)



    def process_sample(self, sample: np.ndarray, dtype=np.float32) -> np.ndarray:
        """Convert the provided 1D audio sample to a 2D spectrogram using the AudioFeatureGenerator

        The generated 2D spectrogram dimensions are calculated as follows::

           sample_length = len(sample) = int(sample_length_ms*sample_rate_hz / 1000)
           window_size_length = int(window_size_ms * sample_rate_hz / 1000)
           window_step_length = int(window_step_ms * sample_rate_hz / 1000)
           height = n_features = (sample_length - window_size_length) // window_step_length + 1
           width = n_channels = AudioFeatureGeneratorSettings.filterbank_n_channels


        The dtype argument specifies the data type of the returned spectrogram.
        This must be one of the following:

        * **uint16**: This the raw value generated by the internal AudioFeatureGenerator library
        * **float32**: This is the uint16 value directly casted to a float32
        * **int8**: This is the int8 value generated by the TFLM "micro features" library.
            Refer to the following for the magic that happens here: `micro_features_generator.cc#L84 <https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/micro_speech/micro_features/micro_features_generator.cc#L84>`_

        Args:
            sample: [sample_length] int16 audio sample
            dtype: Output data type, must be int8, uint16, or float32

        Returns:
            [n_features, n_channels] int8, uint16, or float32  spectrogram
        """
        spectrogram = np.zeros(self._spectrogram_shape, dtype=dtype)
        self._wrapper.process_sample(sample, spectrogram)
        return spectrogram


    def activity_was_detected(self) -> bool:
        """Return if activity was detected in the previously processed sample"""
        return self._wrapper.activity_was_detected()