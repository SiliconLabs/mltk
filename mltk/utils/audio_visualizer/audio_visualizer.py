from typing import Callable
import os
import copy
import logging
import threading
import time
import warnings

import soundfile as sf
import librosa
import numpy as np


import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

from mltk.utils.logger import get_logger
from mltk.utils.path import create_tempdir
from mltk.core import load_mltk_model
from mltk.core.preprocess.audio.parallel_generator import ParallelAudioDataGenerator
from mltk.core.preprocess.audio.audio_feature_generator import AudioFeatureGeneratorSettings

from .settings import SettingsManager

from .playsound import playsound


_visualizer = None

class AudioVisualizer(object):
    
    def __init__(self):
        global _visualizer
        
        self.settings = SettingsManager.instance()
        self.settings.load()
        self.audio = None
        self.audio_file_path = None
        self.audio_index = 0
        self.wave_canvas = None 
        self.spectrogram_canvas = None
        self.logger = get_logger()
        self.update_gui_settings:Callable = None
        self.frontend_settings = AudioFeatureGeneratorSettings()
        self.audio_generator = ParallelAudioDataGenerator(
            frontend_settings=self.frontend_settings
        )
        self.audio_step_direction = 0

        self.running_event = threading.Event()
        self.pause_event = threading.Event()
        self.generate_event = threading.Event()
        self.play_event = threading.Event()
        self.stop_playing_callback = None


        self.running_event.set()
        t = threading.Thread(target=self._generation_thread_loop, name='Spectrogram Generation')
        t.setDaemon(True)
        t.start()

        t = threading.Thread(target=self._play_audio_thread_loop, name='Play Audio')
        t.setDaemon(True)
        t.start()

        
    @staticmethod 
    def instance():
        global _visualizer 
        
        if _visualizer == None:
            _visualizer = AudioVisualizer()
        return _visualizer
    
        
    def load_audio(self, file_path=None):
        self.pause_event.set()
        self.audio_index = 0
        self.audio_file_path = file_path or self.audio_file_path  
        self.audio, self.original_sample_rate = librosa.load(
            file_path, 
            sr=None, 
            mono=True, 
            dtype='float32'
        ) 
        self.pause_event.clear()


    def play_audio(self):
        self.play_event.set()
            
        
    def _play_audio_thread_loop(self):
        while self.running_event.is_set():
            if not self.play_event.wait(0.1):
                continue
            self.play_event.clear()
            audio = self.audio
            if audio is None:
                continue

            try:
                if self.stop_playing_callback is not None:
                    self.stop_playing_callback()
                    self.stop_playing_callback = None

                audio = copy.deepcopy(self.audio)
                sr = self.original_sample_rate

                audio = self._apply_transform(audio, sr)
                tmp_path = os.path.join(create_tempdir(), 'visualizer_audio.wav')
                sf.write(tmp_path, audio, int(self.settings.get('general.sample_rate')))
                self.stop_playing_callback = playsound(tmp_path, block=False)
            except Exception as e:
                self.logger.warning(f'Error while playing sound, err: {e}', exc_info=e)


    def load_model(self, model:str):
        mltk_model = load_mltk_model(model)

        try:
            if not hasattr(mltk_model, 'datagen') or not isinstance(mltk_model.datagen, ParallelAudioDataGenerator):
                raise Exception('Model does not define a ParallelAudioDataGenerator')
        except Exception as e:
            self.logger.error('Failed to load model', exc_info=e)
            return 

        self.pause_event.set()

        self.audio_generator = mltk_model.datagen
        self.frontend_settings = self.audio_generator.frontend_settings

        self.settings.set('general.model', mltk_model.model_specification_path)
        self.settings.set('general.sample_rate', self.frontend_settings.sample_rate_hz)
        self.settings.set('general.sample_length_ms', self.frontend_settings.sample_length_ms)
        self.settings.set('frontend.window_size', self.frontend_settings.window_size_ms)
        self.settings.set('frontend.window_step', self.frontend_settings.window_step_ms)
        self.settings.set('frontend.num_channels', self.frontend_settings.filterbank_n_channels)
        self.settings.set('frontend.upper_band_limit', self.frontend_settings.filterbank_upper_band_limit)
        self.settings.set('frontend.lower_band_limit', self.frontend_settings.filterbank_lower_band_limit)
        self.settings.set('frontend.smoothing_bits', self.frontend_settings.noise_reduction_smoothing_bits)
        self.settings.set('frontend.even_smoothing', self.frontend_settings.noise_reduction_even_smoothing)
        self.settings.set('frontend.odd_smoothing', self.frontend_settings.noise_reduction_odd_smoothing)
        self.settings.set('frontend.min_signal_remaining', self.frontend_settings.noise_reduction_min_signal_remaining)
        self.settings.set('frontend.enable_pcan', self.frontend_settings.pcan_enable)
        self.settings.set('frontend.pcan_strength', self.frontend_settings.pcan_strength)
        self.settings.set('frontend.pcan_offset', self.frontend_settings.pcan_offset)
        self.settings.set('frontend.gain_bits', self.frontend_settings.pcan_gain_bits)
        self.settings.set('frontend.enable_log', self.frontend_settings.log_scale_enable)
        self.settings.set('frontend.scale_shift', self.frontend_settings.log_scale_shift)
        self.settings.set('frontend.enable_noise_reduction', self.frontend_settings.noise_reduction_enable)

        if self.update_gui_settings is not None:
            self.update_gui_settings() # pylint: disable=not-callable

        self.pause_event.clear()
        self.draw()


    def step_direction(self, direction):
        self.audio_step_direction = direction
        self.generate_event.set()


    def reset(self, group):
        if group == 'transform':
            default_values = self.audio_generator.default_transform
            del default_values['bg_noise']
            del default_values['bg_noise_factor']
            del default_values['offset_percentage']
            
        else:
            default_values = {}
        
        for key, value in default_values.items():
            self.settings.set(group + '.' + key, value)
            
        self.settings.save()
        
    
    def draw(self):
        self.generate_event.set()


    def _apply_transform(self, audio, sr, forced=False):
        transform_params = self.audio_generator.default_transform
        
        if forced or self.settings.get('transform.enabled'):
            transform_settings = self.settings.get('transform')
            transform_params.update(transform_settings)
        
        transform_params['offset_percentage'] = 0.5
        
        audio = copy.deepcopy(audio)
        return self.audio_generator.apply_transform(audio, sr, transform_params, whole_sample=True)
    
    
    def _apply_frontend(self, audio, settings):
        self.frontend_settings.sample_rate_hz = settings['general']['sample_rate']
        self.frontend_settings.sample_length_ms = settings['general']['sample_length_ms']
        self.frontend_settings.window_size_ms = settings['frontend']['window_size']
        self.frontend_settings.window_step_ms = settings['frontend']['window_step']
        self.frontend_settings.filterbank_n_channels = settings['frontend']['num_channels']
        self.frontend_settings.filterbank_upper_band_limit = settings['frontend']['upper_band_limit']
        self.frontend_settings.filterbank_lower_band_limit = settings['frontend']['lower_band_limit']
        self.frontend_settings.noise_reduction_smoothing_bits = settings['frontend']['smoothing_bits']
        self.frontend_settings.noise_reduction_even_smoothing = settings['frontend']['even_smoothing']
        self.frontend_settings.noise_reduction_odd_smoothing = settings['frontend']['odd_smoothing']
        self.frontend_settings.noise_reduction_min_signal_remaining = settings['frontend']['min_signal_remaining']
        self.frontend_settings.pcan_enable = settings['frontend']['enable_pcan']
        self.frontend_settings.pcan_strength = settings['frontend']['pcan_strength']
        self.frontend_settings.pcan_offset = settings['frontend']['pcan_offset']
        self.frontend_settings.pcan_gain_bits = settings['frontend']['gain_bits']
        self.frontend_settings.log_scale_enable = settings['frontend']['enable_log']
        self.frontend_settings.log_scale_shift = settings['frontend']['scale_shift']
        self.frontend_settings.noise_reduction_enable = settings['frontend']['enable_noise_reduction']

        sample_length_seconds = self.frontend_settings.sample_length_ms/1000
        sample_length = int(sample_length_seconds * self.frontend_settings.sample_rate_hz)
        if len(audio) < sample_length:
            audio = np.concatenate((audio, np.zeros((sample_length - len(audio),), dtype=audio.dtype)), axis=0) 
        elif len(audio) > sample_length:
            audio = audio[:sample_length]
            
        return self.audio_generator.apply_frontend(audio)


    def _generation_thread_loop(self):
        while self.running_event.is_set():
            pipeline_enabled = self.settings.get('pipeline.enabled')
            pipeline_period = self.settings.get('pipeline.period_ms')/1000

            is_set = self.generate_event.wait(pipeline_period)
            self.generate_event.clear()
            if self.pause_event.is_set():
                continue
            if is_set or pipeline_enabled:
                try:
                    self._generate()
                except Exception as e:
                    self.logger.warning(f'Error while generating spectrogram, err: {e}', exc_info=e)


    def _get_next_audio_chunk(self, settings):
        if self.audio is None:
            return None 
        
        pipeline_enabled = settings['pipeline']['enabled'] 
        pipeline_period_seconds = settings['pipeline']['period_ms']/1000
        pipeline_length = int(pipeline_period_seconds * self.original_sample_rate)
        sample_length_ms = settings['general']['sample_length_ms']
        sample_length_seconds = sample_length_ms/1000
        sample_length = int(sample_length_seconds * self.original_sample_rate)
        audio_length = len(self.audio)
        audio_chunk = np.ndarray((sample_length,), dtype=np.float32)

        direction = self.audio_step_direction 
        self.audio_step_direction = 0 

        if direction == -1:
            if self.audio_index == 0:
                self.audio_index = audio_length - pipeline_length
            else:
                self.audio_index -= pipeline_length
            self.logger.debug(f'Audio index: {self.audio_index}')
        elif direction == 1:
            self.audio_index = (self.audio_index + pipeline_length) % audio_length
            self.logger.debug(f'Audio index: {self.audio_index}')
        elif pipeline_enabled:
            self.audio_index = (self.audio_index + pipeline_length) % audio_length

        audio_start_index = self.audio_index
        chunk_start_index = 0
      
        remaining_length = sample_length
        while remaining_length > 0:
            audio_length_to_end = audio_length - audio_start_index
            copy_length = min(audio_length_to_end, remaining_length)
            audio_chunk[chunk_start_index:chunk_start_index+copy_length] = self.audio[audio_start_index : audio_start_index + copy_length]
            remaining_length -= copy_length
            audio_start_index = (audio_start_index + copy_length) % audio_length
            chunk_start_index += copy_length

        return audio_chunk
    

    def _generate(self):
        settings = self.settings.get_snapshot()

        audio_chunk = self._get_next_audio_chunk(settings)
        if audio_chunk is None:
            return

        if self.audio_generator.noaug_preprocessing_function is not None:
            audio_chunk = self.audio_generator.noaug_preprocessing_function(None, audio_chunk)

        audio_chunk = self._apply_transform(audio_chunk, self.original_sample_rate)
        
        frontend_enabled = settings['frontend']['enabled']
        
        kwargs = {
            'precomputed'   : frontend_enabled,
            'axis_labels'   : True,
            'size_label'    : True,
            'sr'            : settings['general']['sample_rate'],
            'n_mels'        : settings['frontend']['num_channels'],
            'window_size_ms': settings['frontend']['window_size'],
            'hop_length_ms' : settings['frontend']['window_step'],
            'fmax'          : settings['frontend']['upper_band_limit'],
            'fmin'          : settings['frontend']['lower_band_limit'],
        }
        
        if self.audio_generator.preprocessing_function is not None:
            audio_chunk = self.audio_generator.preprocessing_function(None, audio_chunk)

        if frontend_enabled:
            audio_chunk = self._apply_frontend(audio_chunk, settings)

        if not self.spectrogram_canvas is None:
            self.spectrogram_canvas.plot(audio_chunk, **kwargs)
