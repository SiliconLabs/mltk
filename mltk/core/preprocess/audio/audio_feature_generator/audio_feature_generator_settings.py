from __future__ import annotations
from typing import Tuple
import copy



class AudioFeatureGeneratorSettings(dict):
    """Settings for the `AudioFeatureGenerator <https://siliconlabs.github.io/mltk/docs/python_api/data_preprocessing/audio_feature_generator.html>`_


    **Example Usage**

    .. highlight:: python
    .. code-block:: python

        from mltk.core.preprocess.audio.audio_feature_generator import AudioFeatureGeneratorSettings

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

        # If this is used in a model specification file,
        # be sure to add the Audio Feature generator settings to the model parameters.
        # This way, they are included in the generated .tflite model file
        # See https://siliconlabs.github.io/mltk/docs/guides/model_parameters.html
        my_model.model_parameters.update(frontend_settings)


    See the `Audio Feature Generator <https://siliconlabs.github.io/mltk/docs/audio/audio_feature_generator.html>`_ guide for more details.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.sample_rate_hz = 16000
        self.sample_length_ms = 1000
        self.window_size_ms = 25
        self.window_step_ms = 10

        self.filterbank_n_channels = 32
        self.filterbank_upper_band_limit = 7500.0
        self.filterbank_lower_band_limit = 125.0
        self.noise_reduction_enable = False
        self.noise_reduction_smoothing_bits = 10
        self.noise_reduction_even_smoothing = 0.025
        self.noise_reduction_odd_smoothing =  0.06
        self.noise_reduction_min_signal_remaining = 0.05
        self.pcan_enable = False
        self.pcan_strength = 0.95
        self.pcan_offset = 80.0
        self.pcan_gain_bits = 21
        self.log_scale_enable = True
        self.log_scale_shift = 6
        self.activity_detection_enable = False
        self.activity_detection_alpha_a = 0.5
        self.activity_detection_alpha_b = 0.8
        self.activity_detection_arm_threshold = 0.75
        self.activity_detection_trip_threshold = 0.8
        self.dc_notch_filter_enable = False
        self.dc_notch_filter_coefficient = 0.95
        self.quantize_dynamic_scale_enable = False
        self.quantize_dynamic_scale_range_db = 40.0

        # Update the dict with the given values
        # AFTER setting the defaults
        super().update(kwargs)



    @property
    def spectrogram_shape(self) -> Tuple[int, int]:
        """Return the generated spectrogram shape as (height, width) i.e. (n_features, filterbank_n_channels)"""
        window_size_length = int((self.window_size_ms * self.sample_rate_hz) / 1000)
        window_step_length = int((self.window_step_ms * self.sample_rate_hz) / 1000)
        height = (self.sample_length - window_size_length) // window_step_length + 1
        width = self.filterbank_n_channels
        return (height, width)

    @property
    def sample_rate_hz(self) -> int:
        """The sample rate of the audio in Hz, default 16000"""
        return self.get('fe.sample_rate_hz', 0)
    @sample_rate_hz.setter
    def sample_rate_hz(self, v: int):
        s = int(v)
        if s <= 0 or s > 10e6:
            raise ValueError(f'Invalid sample_rate_hz, {v}')
        self['fe.sample_rate_hz'] = s
        self._update_fft_length()

    @property
    def sample_length_ms(self) -> int:
        """The length of an audio sample in milliseconds, default 1000"""
        return self['fe.sample_length_ms']
    @sample_length_ms.setter
    def sample_length_ms(self, v: int):
        s = int(v)
        if s <= 0 or s > 10e6:
            msg = ''
            if isinstance(v, float) and v < 1:
                msg = '. You may need to multiply this value by 1000'
            raise ValueError(f'Invalid sample_length_ms, {v}{msg}')
        self['fe.sample_length_ms'] = s

    @property
    def sample_length(self) -> int:
        """Calculated length of an audio sample in frames
        sample_length = (self.sample_length_ms * self.sample_rate_hz) // 1000
        """
        sample_length      = int((self.sample_length_ms * self.sample_rate_hz) / 1000)
        return sample_length

    @property
    def window_size_ms(self) -> int:
        """length of desired time frames in ms, default 25"""
        return self.get('fe.window_size_ms', 0)
    @window_size_ms.setter
    def window_size_ms(self, v: int):
        s = int(v)
        if s <= 0 or s > 10e6:
            msg = ''
            if isinstance(v, float) and v < 1:
                msg = '. You may need to multiply this value by 1000'
            raise ValueError(f'Invalid window_size_ms, {v}{msg}')
        self['fe.window_size_ms'] = s
        self._update_fft_length()

    @property
    def window_step_ms(self) -> int:
        """length of step size for the next frame in ms, default 10"""
        return self['fe.window_step_ms']
    @window_step_ms.setter
    def window_step_ms(self, v: int):
        s = int(v)
        if s <= 0 or s > 10e6:
            msg = ''
            if isinstance(v, float) and v < 1:
                msg = '. You may need to multiply this value by 1000'
            raise ValueError(f'Invalid window_step_ms, {v}{msg}')
        self['fe.window_step_ms'] = s

    @property
    def filterbank_n_channels(self) -> int:
        """the number of filterbank channels to use, default 32"""
        return self['fe.filterbank_n_channels']
    @filterbank_n_channels.setter
    def filterbank_n_channels(self, v: int):
        s = int(v)
        if s <= 0 or s > 10e6:
            raise ValueError(f'Invalid filterbank_n_channels, {v}')
        self['fe.filterbank_n_channels'] = s

    @property
    def filterbank_upper_band_limit(self) -> float:
        """ Float, the highest frequency included in the filterbanks, default 7500.0
        NOTE: This should be no more than sample_rate_hz / 2
        """
        return self['fe.filterbank_upper_band_limit']
    @filterbank_upper_band_limit.setter
    def filterbank_upper_band_limit(self, v: float):
        self['fe.filterbank_upper_band_limit'] = float(v)

    @property
    def filterbank_lower_band_limit(self) -> float:
        """ the lowest frequency included in the filterbanks, default 125.0"""
        return self['fe.filterbank_lower_band_limit']
    @filterbank_lower_band_limit.setter
    def filterbank_lower_band_limit(self, v: float):
        self['fe.filterbank_lower_band_limit'] = float(v)

    @property
    def noise_reduction_enable(self) -> bool:
        """Enable/disable noise reduction module, default false"""
        return self['fe.noise_reduction_enable']
    @noise_reduction_enable.setter
    def noise_reduction_enable(self, v: bool):
        self['fe.noise_reduction_enable'] = bool(v)

    @property
    def noise_reduction_smoothing_bits(self) -> int:
        """scale up signal by 2^(smoothing_bits) before reduction, default 10"""
        return self['fe.noise_reduction_smoothing_bits']
    @noise_reduction_smoothing_bits.setter
    def noise_reduction_smoothing_bits(self, v: int):
        self['fe.noise_reduction_smoothing_bits'] = int(v)

    @property
    def noise_reduction_even_smoothing(self) -> float:
        """smoothing coefficient for even-numbered channels, default 0.025"""
        return self['fe.noise_reduction_even_smoothing']
    @noise_reduction_even_smoothing.setter
    def noise_reduction_even_smoothing(self, v: float):
        self['fe.noise_reduction_even_smoothing'] = float(v)

    @property
    def noise_reduction_odd_smoothing(self) -> float:
        """smoothing coefficient for odd-numbered channels, default 0.06"""
        return self['fe.noise_reduction_odd_smoothing']
    @noise_reduction_odd_smoothing.setter
    def noise_reduction_odd_smoothing(self, v: float):
        self['fe.noise_reduction_odd_smoothing'] = float(v)

    @property
    def noise_reduction_min_signal_remaining(self) -> float:
        """fraction of signal to preserve in smoothing, default 0.05"""
        return self['fe.noise_reduction_min_signal_remaining']
    @noise_reduction_min_signal_remaining.setter
    def noise_reduction_min_signal_remaining(self, v: float):
        self['fe.noise_reduction_min_signal_remaining'] = float(v)

    @property
    def pcan_enable(self) -> bool:
        """enable PCAN auto gain control, default false"""
        return self['fe.pcan_enable']
    @pcan_enable.setter
    def pcan_enable(self, v: bool):
        self['fe.pcan_enable'] = bool(v)

    @property
    def pcan_strength(self) -> float:
        """ gain normalization exponent, default 0.95"""
        return self['fe.pcan_strength']
    @pcan_strength.setter
    def pcan_strength(self, v: float):
        self['fe.pcan_strength'] = float(v)

    @property
    def pcan_offset(self) -> float:
        """positive value added in the normalization denominator, default 80.0"""
        return self['fe.pcan_offset']
    @pcan_offset.setter
    def pcan_offset(self, v: float):
        self['fe.pcan_offset'] = float(v)

    @property
    def pcan_gain_bits(self) -> int:
        """number of fractional bits in the gain, default 21"""
        return self['fe.pcan_gain_bits']
    @pcan_gain_bits.setter
    def pcan_gain_bits(self, v: int):
        self['fe.pcan_gain_bits'] = int(v)

    @property
    def log_scale_enable(self) -> bool:
        """enable logarithmic scaling of filterbanks, default true"""
        return self['fe.log_scale_enable']
    @log_scale_enable.setter
    def log_scale_enable(self, v: bool):
        self['fe.log_scale_enable'] = bool(v)

    @property
    def log_scale_shift(self) -> int:
        """scale filterbanks by 2^(scale_shift), default 6"""
        return self['fe.log_scale_shift']
    @log_scale_shift.setter
    def log_scale_shift(self, v: int):
        self['fe.log_scale_shift'] = int(v)

    @property
    def activity_detection_enable(self) -> bool:
        """Enable the activity detection block.
        This indicates when activity, such as a speech command, is detected in the audio stream,
        default False"""
        return self['fe.activity_detection_enable']
    @activity_detection_enable.setter
    def activity_detection_enable(self, v: bool):
        self['fe.activity_detection_enable'] = bool(v)

    @property
    def activity_detection_alpha_a(self) -> float:
        """Activity detection filter A coefficient
        The activity detection "fast filter" coefficient.
        The filter is a 1-real pole IIR filter: ``computes out = (1-k)*in + k*out``
        Default 0.5"""
        return self['fe.activity_detection_alpha_a']
    @activity_detection_alpha_a.setter
    def activity_detection_alpha_a(self, v: float):
        self['fe.activity_detection_alpha_a'] = float(v)

    @property
    def activity_detection_alpha_b(self) -> float:
        """Activity detection filter B coefficient
        The activity detection "slow filter" coefficient.
        The filter is a 1-real pole IIR filter: ``computes out = (1-k)*in + k*out``
        Default 0.8"""
        return self['fe.activity_detection_alpha_b']
    @activity_detection_alpha_b.setter
    def activity_detection_alpha_b(self, v: float):
        self['fe.activity_detection_alpha_b'] = float(v)

    @property
    def activity_detection_arm_threshold(self) -> float:
        """Threshold for arming the detection block
        The threshold for when there should be considered possible activity in the audio stream
        Default 0.75"""
        return self['fe.activity_detection_arm_threshold']
    @activity_detection_arm_threshold.setter
    def activity_detection_arm_threshold(self, v: float):
        self['fe.activity_detection_arm_threshold'] = float(v)

    @property
    def activity_detection_trip_threshold(self) -> float:
        """Threshold for tripping the detection block
        The threshold for when activity is considered detected in the audio stream
        Default 0.8"""
        return self['fe.activity_detection_trip_threshold']
    @activity_detection_trip_threshold.setter
    def activity_detection_trip_threshold(self, v: float):
        self['fe.activity_detection_trip_threshold'] = float(v)

    @property
    def dc_notch_filter_enable(self) -> bool:
        """Enable the DC notch filter
        This will help negate any DC components in the audio signal
        Default False"""
        return self['fe.dc_notch_filter_enable']
    @dc_notch_filter_enable.setter
    def dc_notch_filter_enable(self, v: bool):
        self['fe.dc_notch_filter_enable'] = bool(v)

    @property
    def dc_notch_filter_coefficient(self) -> float:
        """Coefficient used by DC notch filter

        The DC notch filter coefficient k in Q(16,15) format, ``H(z) = (1 - z^-1)/(1 - k*z^-1)``
        Default 0.95"""
        return self['fe.dc_notch_filter_coefficient']
    @dc_notch_filter_coefficient.setter
    def dc_notch_filter_coefficient(self, v: float):
        self['fe.dc_notch_filter_coefficient'] = float(v)

    @property
    def quantize_dynamic_scale_enable(self) -> bool:
        """Enable dynamic quantization

        Enable dynamic quantization of the generated audio spectrogram.
        With this, the max spectrogram value is mapped to +127,
        and the max spectrogram minus :py:class:`~quantize_dynamic_scale_range_db` is mapped to -128.
        Anything below max spectrogram minus :py:class:`~quantize_dynamic_scale_range_db` is mapped to -128.
        Default False"""
        return self['fe.quantize_dynamic_scale_enable']
    @quantize_dynamic_scale_enable.setter
    def quantize_dynamic_scale_enable(self, v: bool):
        self['fe.quantize_dynamic_scale_enable'] = bool(v)

    @property
    def quantize_dynamic_scale_range_db(self) -> float:
        """Rhe dynamic range in dB used by the dynamic quantization, default 40.0"""
        return self['fe.quantize_dynamic_scale_range_db']
    @quantize_dynamic_scale_range_db.setter
    def quantize_dynamic_scale_range_db(self, v: float):
        self['fe.quantize_dynamic_scale_range_db'] = float(v)


    @property
    def fft_length(self) -> int:
        """The calculated size required to do an FFT.
        This is dependent on the window_size_ms and sample_rate_hz values"""
        return self['fe.fft_length']


    def copy(self) -> AudioFeatureGeneratorSettings:
        """Return a deep copy of the current settings"""
        return copy.deepcopy(self)

    def _update_fft_length(self):
        windows_size = int((self.window_size_ms * self.sample_rate_hz) / 1000)
        # The FFT length is the smallest power of 2 that
        # is larger than the window size
        fft_length = 1
        while fft_length < windows_size:
            fft_length <<= 1
        self['fe.fft_length'] = fft_length