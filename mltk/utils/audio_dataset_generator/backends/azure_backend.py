"""
Microsoft "Azure" Text-to-Speech Backend

Install     : pip install azure-cognitiveservices-speech
Quick start : https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/get-started-text-to-speech?pivots=programming-language-python
Online demo : https://azure.microsoft.com/en-us/products/cognitive-services/text-to-speech/#features
Pricing     : https://azure.microsoft.com/en-us/pricing/details/cognitive-services/speech-services
Quotas      : https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/speech-services-quotas-and-limits
Docs        : https://learn.microsoft.com/en-us/python/api/azure-cognitiveservices-speech/azure.cognitiveservices.speech.speechsynthesizer?view=azure-python
"""
from __future__ import annotations
from typing import List
from dataclasses import dataclass
import os
from mltk.utils.python import install_pip_package




from ..generator_types import (
    BackendBase,
    Augmentation,
    Voice,
    GenerationConfig,
    logger,
)


class AzureBackend(BackendBase):
    """Microsoft Azure Backend

    See: https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/get-started-text-to-speech?pivots=programming-language-python
    """
    def __init__(self):
        super().__init__(
            transactions_per_second=200 # 200 transactions per second
        )
        self._config = None


    @property
    def name(self) -> str:
        return 'azure'


    def load(self, install_python_package=False, **kwargs):
        if install_python_package:
            install_pip_package('azure-cognitiveservices-speech', 'azure.cognitiveservices.speech', logger=logger)

        try:
            # pip install azure-cognitiveservices-speech
            import azure.cognitiveservices.speech as speechsdk
        except ModuleNotFoundError:
            raise RuntimeError('To use the Azure backend, first run the command:\npip install azure-cognitiveservices-speech')


        subscription = kwargs.pop('subscription', os.environ.get('SPEECH_KEY', None))
        region = kwargs.pop('region', os.environ.get('SPEECH_REGION', None))

        try:
            self._config = speechsdk.SpeechConfig(subscription=subscription, region=region, **kwargs)
            self._config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm)
            self.list_languages()
        except Exception as e: # pylint:disable=redefined-outer-name
            raise RuntimeError(f'\n\nFailed to load Microsoft backend, err: {e}\n\nFor more details see: https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/get-started-text-to-speech?pivots=programming-language-python')


    def list_languages(self) -> List[str]:
        import azure.cognitiveservices.speech as speechsdk

        languages = set()
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=self._config, audio_config=None)
        result = speech_synthesizer.get_voices_async().get()
        if result.reason != speechsdk.ResultReason.VoicesListRetrieved:
            raise RuntimeError(f'{result.reason}: {result.error_details}')

        for voice in result.voices:
            name = voice.short_name
            lang = '-'.join(name.split('-')[:2])
            languages.add(lang)

        return list(languages)


    def list_voices(self, language_code:str=None) -> List[AzureVoice]:
        import azure.cognitiveservices.speech as speechsdk

        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=self._config, audio_config=None)
        result = speech_synthesizer.get_voices_async().get()
        if result.reason != speechsdk.ResultReason.VoicesListRetrieved:
            raise RuntimeError(f'{result.reason}: {result.error_details}')


        retval = []
        for voice in result.voices:
            name = voice.short_name
            lang = '-'.join(name.split('-')[:2])
            if language_code and language_code != lang:
                continue
            style_list = None if not voice.style_list[0] else voice.style_list
            retval.append(AzureVoice(
                backend=self.name,
                name=voice.short_name,
                language_code=lang,
                styles=style_list
            ))

        return retval


    def list_configurations(
        self,
        augmentations:List[Augmentation],
        voice:AzureVoice,
    ) -> List[GenerationConfig]:
        assert isinstance(voice,AzureVoice)
        retval:List[GenerationConfig] = []
        styles = voice.styles or [None]
        for style in styles:
            for aug in augmentations:
                retval.append(AzureGenerationConfig(
                    voice=voice,
                    style=style,
                    rate=aug.rate,
                    pitch=aug.pitch,
                    keyword=None
                ))
        return retval


    def count_characters(self, config:AzureGenerationConfig) -> int:
        assert config.voice.backend == self.name
        text = self._generate_msg(config)
        return len(text)

    def generate_filename(self, config:AzureGenerationConfig) -> str:
        assert config.voice.backend == self.name
        return f'{self.name}_{config.voice.language_code}+{config.voice.name}+{config.style}+{config.keyword}+{config.rate}+{config.pitch}+{config.voice.hex_hash}.wav'

    def generate(self, config:AzureGenerationConfig, out_dir:str) -> str:
        import azure.cognitiveservices.speech as speechsdk
        assert config.voice.backend == self.name

        out_path = f'{out_dir}/{self.generate_filename(config)}'
        if os.path.exists(out_path):
            import time
            time.sleep(.250)
            return out_path

        ssml_msg = self._generate_msg(config)
        output_config = speechsdk.audio.AudioOutputConfig(filename=out_path)

        speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self._config,
            audio_config=output_config
        )

        try:
            self.update_generate_timestamp()
            result = speech_synthesizer.speak_ssml(ssml_msg)
            if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
                if result.reason == speechsdk.ResultReason.Canceled:
                    cancellation_details = result.cancellation_details
                    raise RuntimeError(f'{cancellation_details.reason}: {cancellation_details.error_details}')
                else:
                    raise RuntimeError(f'{result.reason}: {result}')

            elif os.path.getsize(out_path) == 0:
                raise RuntimeError('Empty file generated')
        except Exception as e: # pylint:disable=redefined-outer-name
            try:
                os.remove(out_path)
            except:
                pass
            raise RuntimeError(f'{self.name} backend: Failed to generate: {out_path}, err: {e}')

        return out_path


    def _generate_msg(self, config:AzureGenerationConfig) -> str:
        rate = config.rate
        pitch = config.pitch
        style = config.style

        if config.rate in ('default', 'medium'):
            rate = None

        if pitch in ('default', 'medium'):
            pitch = None

        ssml_msg  = f'<speak version="1.0" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="{config.voice.language_code}">'
        ssml_msg += f'<voice name="{config.voice.name}">'
        if style:
            ssml_msg += f'<mstts:express-as style="{style}">'

        if pitch or rate:
            ssml_msg += '<prosody'
        if pitch:
            ssml_msg += f' pitch="{pitch}"'
        if rate:
            ssml_msg += f' rate="{rate}"'
        if pitch or rate:
            ssml_msg += '>'

        ssml_msg += config.keyword

        if pitch or rate:
            ssml_msg += '</prosody>'

        if style:
            ssml_msg += '</mstts:express-as>'

        ssml_msg += '</voice>'
        ssml_msg += '</speak>'

        return ssml_msg



@dataclass
class AzureVoice(Voice):
    styles:List[str]=None

    def __hash__(self):
        if self.styles:
            return hash((self.name, self.language_code, self.backend, *self.styles))
        else:
            return super().__hash__()


@dataclass
class AzureGenerationConfig(GenerationConfig):
    style:str=None



