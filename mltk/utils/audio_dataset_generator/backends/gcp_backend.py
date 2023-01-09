"""
Google Cloud Platform (GCP) Text-to-Speech Backend

Install     : pip install grpcio-status==1.48.* google-cloud-texttospeech
Quick start : https://codelabs.developers.google.com/codelabs/cloud-text-speech-python3
Online demo : https://cloud.google.com/text-to-speech#section-2
Pricing     : https://cloud.google.com/text-to-speech/pricing
Quotas      : https://cloud.google.com/text-to-speech/quotas
Docs        : https://cloud.google.com/python/docs/reference/texttospeech/latest/google.cloud.texttospeech_v1beta1.services.text_to_speech.TextToSpeechClient
"""
from __future__ import annotations
from typing import List
from dataclasses import dataclass
import os
from mltk.utils.python import install_pip_package

try:
    import google.cloud.texttospeech as tts
    _have_tts = True
except ModuleNotFoundError:
    _have_tts = False


from ..generator_types import (
    BackendBase,
    Augmentation,
    Voice,
    GenerationConfig,
    VoicePitch,
    VoiceRate,
    logger
)


class GcpBackend(BackendBase):
    """Google Cloud Platform Backend

    See: https://codelabs.developers.google.com/codelabs/cloud-text-speech-python3
    """
    def __init__(self) -> None:
        super().__init__(
            transactions_per_second=1000/60
        )
        self._client = None


    @property
    def name(self) -> str:
        return 'gcp'

    def load(self, install_python_package=False, **kwargs):
        if not _have_tts and install_python_package:
            install_pip_package('grpcio-status==1.48.*', logger=logger)
            install_pip_package('google-cloud-texttospeech', logger=logger)

        try:
            # pip install grpcio-status==1.48.* google-cloud-texttospeech
            import google.cloud.texttospeech as tts
        except ModuleNotFoundError:
            raise RuntimeError('To use the Google Cloud Platform backend, first run the command:\npip install grpcio-status==1.48.* google-cloud-texttospeech')

        try:
            self._client = tts.TextToSpeechClient(**kwargs)
            self.list_languages()
        except Exception as e: # pylint:disable=redefined-outer-name
            raise RuntimeError(f'\n\nFailed to load Google backend, err: {e}\n\nFor more details see: https://codelabs.developers.google.com/codelabs/cloud-text-speech-python3')


    def list_languages(self) -> List[str]:
        response = self._client.list_voices()
        languages = set()
        for voice in response.voices:
            for language_code in voice.language_codes:
                languages.add(language_code)

        return list(languages)


    def list_voices(self, language_code:str=None) -> List[GcpVoice]:
        retval:List[GcpVoice] = []
        response = self._client.list_voices(language_code=language_code)
        for voice in response.voices:
            for lang in voice.language_codes:
                retval.append(GcpVoice(
                    backend=self.name,
                    name=voice.name,
                    language_code=lang,
                ))

        return retval


    def list_configurations(
        self,
        augmentations:List[Augmentation],
        voice:GcpVoice,
    ) -> List[GenerationConfig]:
        assert isinstance(voice,GcpVoice)

        retval:List[GenerationConfig] = []

        for aug in augmentations:
            retval.append(GenerationConfig(
                voice=voice,
                rate=aug.rate,
                pitch=aug.pitch,
                keyword=None
            ))

        return retval


    def count_characters(self, config:GenerationConfig) -> int:
        assert config.voice.backend == self.name
        return len(config.keyword)


    def generate_filename(self, config:GenerationConfig) -> str:
        assert config.voice.backend == self.name
        return f'{self.name}_{config.voice.language_code}+{config.voice.name}+{config.keyword}+{config.rate}+{config.pitch}+{config.voice.hex_hash}.wav'


    def generate(self, config:GenerationConfig, out_dir:str) -> str:
        import google.cloud.texttospeech as tts
        assert config.voice.backend == self.name
        out_path = f'{out_dir}/{self.generate_filename(config)}'
        if os.path.exists(out_path):
            import time
            time.sleep(.250)
            return out_path


        text_input = tts.SynthesisInput(text=config.keyword)
        voice_params = tts.VoiceSelectionParams(
            language_code=config.voice.language_code,
            name=config.voice.name
        )
        audio_config = tts.AudioConfig(
            audio_encoding=tts.AudioEncoding.LINEAR16,
            speaking_rate = voice_rate_to_float(config.rate),
            pitch = voice_pitch_to_float(config.pitch),
            sample_rate_hertz=16000,
        )

        try:
            self.update_generate_timestamp()
            response = self._client.synthesize_speech(
                input=text_input,
                voice=voice_params,
                audio_config=audio_config
            )
            with open(out_path, "wb") as out:
                out.write(response.audio_content)

        except Exception as e: # pylint:disable=redefined-outer-name
            try:
                os.remove(out_path)
            except:
                pass
            raise RuntimeError(f'{self.name} backend: Failed to generate: {out_path}, err: {e}')

        return out_path


def voice_rate_to_float(rate:VoiceRate) -> float:
    # https://cloud.google.com/text-to-speech#section-2
    if rate == VoiceRate.xslow:
        return 0.5
    if rate in (VoiceRate.medium, VoiceRate.default):
        return 1.0
    if rate in VoiceRate.xfast:
        return 2.0


def voice_pitch_to_float(pitch:VoicePitch) -> float:
    # https://cloud.google.com/text-to-speech#section-2
    if pitch == VoicePitch.low:
        return -3.0
    if pitch in (VoicePitch.medium, VoicePitch.default):
        return 0.0
    if pitch in VoicePitch.high:
        return 3.0



@dataclass
class GcpVoice(Voice):
    def __hash__(self): # pylint:disable=useless-parent-delegation
        return super().__hash__()

