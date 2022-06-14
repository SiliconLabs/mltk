from typing import Tuple




class AudioFeatureGeneratorSettings(dict):
    """AudioFeatureGenerator Settings

    See the `Audio Feature Generator <https://siliconlabs.github.io/mltk/docs/audio/audio_feature_generator.html>`_ guide for more details.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self['fe.sample_rate_hz'] = 16000
        self['fe.sample_length_ms'] = 1000
        self['fe.window_size_ms'] = 25
        self['fe.window_step_ms'] = 10

        self['fe.filterbank_n_channels'] = 32
        self['fe.filterbank_upper_band_limit'] = 7500.0
        self['fe.filterbank_lower_band_limit'] = 125.0
        self['fe.noise_reduction_enable'] = False
        self['fe.noise_reduction_smoothing_bits'] = 10
        self['fe.noise_reduction_even_smoothing'] = 0.025
        self['fe.noise_reduction_odd_smoothing'] =  0.06
        self['fe.noise_reduction_min_signal_remaining'] = 0.05
        self['fe.pcan_enable'] = False
        self['fe.pcan_strength'] = 0.95
        self['fe.pcan_offset'] = 80.0
        self['fe.pcan_gain_bits'] = 21
        self['fe.log_scale_enable'] = True
        self['fe.log_scale_shift'] = 6
        self['fe.activity_detection_enable'] = False
        self['fe.activity_detection_alpha_a'] = 0.5
        self['fe.activity_detection_alpha_b'] = 0.8
        self['fe.activity_detection_arm_threshold'] = 0.75
        self['fe.activity_detection_trip_threshold'] = 0.8
        self['fe.dc_notch_filter_enable'] = False
        self['fe.dc_notch_filter_coefficient'] = 0.95
        self['fe.quantize_dynamic_scale_enable'] = False
        self['fe.quantize_dynamic_scale_range_db'] = 40.0



    @property
    def spectrogram_shape(self) -> Tuple[int, int]:
        """Return the generated spectrogram shape as (height, width) i.e. (n_features, filterbank_n_channels)"""
        window_size_length = int(self.window_size_ms * self.sample_rate_hz / 1000)
        window_step_length = int(self.window_step_ms * self.sample_rate_hz / 1000)
        sample_length      = int(self.sample_length_ms * self.sample_rate_hz / 1000)
        height = (sample_length - window_size_length) // window_step_length + 1
        width = self.filterbank_n_channels
        return (height, width)

    @property
    def sample_rate_hz(self) -> int:
        """The sample rate of the audio in Hz, default 16000"""
        return self['fe.sample_rate_hz']
    @sample_rate_hz.setter
    def sample_rate_hz(self, v: int):
        self['fe.sample_rate_hz'] = int(v)

    @property
    def sample_length_ms(self) -> int:
        """The length of an audio sample in milliseconds, default 1000"""
        return self['fe.sample_length_ms']
    @sample_length_ms.setter
    def sample_length_ms(self, v: int):
        self['fe.sample_length_ms'] = int(v)

    @property
    def window_size_ms(self) -> int:
        """length of desired time frames in ms, default 25"""
        return self['fe.window_size_ms']
    @window_size_ms.setter
    def window_size_ms(self, v: int):
        self['fe.window_size_ms'] = int(v)

    @property
    def window_step_ms(self) -> int:
        """length of step size for the next frame in ms, default 10"""
        return self['fe.window_step_ms']
    @window_step_ms.setter
    def window_step_ms(self, v: int):
        self['fe.window_step_ms'] = int(v)

    @property
    def filterbank_n_channels(self) -> int:
        """the number of filterbank channels to use, default 32"""
        return self['fe.filterbank_n_channels']
    @filterbank_n_channels.setter
    def filterbank_n_channels(self, v: int):
        self['fe.filterbank_n_channels'] = int(v)

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