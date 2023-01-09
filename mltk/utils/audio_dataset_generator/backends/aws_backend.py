"""
Amazon "Polly" Text-to-Speech Backend

Install     : pip install boto3
Quick start : https://docs.aws.amazon.com/polly/latest/dg/get-started-what-next.html
Online demo : https://us-east-1.console.aws.amazon.com/polly/home/SynthesizeSpeech
Pricing     : https://aws.amazon.com/polly/pricing
Quotas      : https://docs.aws.amazon.com/general/latest/gr/pol.html
Docs        : https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/polly.html

"""
from __future__ import annotations
from typing import List, Tuple
from dataclasses import dataclass
import contextlib
import wave
import os
from mltk.utils.python import install_pip_package



from ..generator_types import (
    BackendBase,
    Augmentation,
    Voice,
    GenerationConfig,
    VoicePitch,
    logger
)


class AwsBackend(BackendBase):
    """Amazon Web Services Backend

    See https://docs.aws.amazon.com/polly/latest/dg/get-started-what-next.html
    """
    def __init__(self) -> None:
        super().__init__(
            transactions_per_second=8/1
        )
        self._client = None


    @property
    def name(self) -> str:
        return 'aws'


    def load(self, install_python_package=False, **kwargs):
        if install_python_package:
            install_pip_package('boto3', logger=logger)

        try:
            # pip install boto3
            import boto3
            from boto3 import Session
        except ModuleNotFoundError:
            raise RuntimeError('To use the AWS backend, first run the command:\npip install boto3')

        try:
            session = Session(**kwargs)
            self._client = session.client("polly")
            self.list_languages()
        except Exception as e: # pylint:disable=redefined-outer-name
            raise RuntimeError(f'\n\nFailed to load AWS backend, err: {e}\n\nFor more details see: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html')


    def list_languages(self) -> List[str]:
        languages = set()
        params = {}
        while True:
            response = self._client.describe_voices(**params)
            for voice in response['Voices']:
                languages.add(voice['LanguageCode'])
            if "NextToken" in response:
                params = {"NextToken": response["NextToken"]}
            else:
                break

        return list(languages)


    def list_voices(self, language_code:str=None) -> List[AwsVoice]:
        kwargs = {}
        if language_code:
            kwargs['LanguageCode'] = language_code
        kwargs['IncludeAdditionalLanguageCodes'] = True

        while True:
            response = self._client.describe_voices(**kwargs)
            voices = sorted(response['Voices'], key=lambda voice: voice['Id'])

            retval:List[AwsVoice] = []
            for voice in voices:
                for engine in voice['SupportedEngines']:
                    retval.append(AwsVoice(
                        backend=self.name,
                        name=voice['Id'],
                        language_code=voice['LanguageCode'],
                        engine=engine
                        ))
                    for lang in voice.get('AdditionalLanguageCodes', []):
                        retval.append(AwsVoice(
                            backend=self.name,
                            name=voice['Id'],
                            language_code=lang,
                            engine=engine
                        ))

            if "NextToken" in response:
                kwargs['NextToken'] = response['NextToken']
            else:
                break

        return retval

    def list_configurations(
        self,
        augmentations:List[Augmentation],
        voice:AwsVoice,
    ) -> List[GenerationConfig]:
        assert isinstance(voice,AwsVoice)

        retval:List[GenerationConfig] = []

        for aug in augmentations:
            if aug.pitch != VoicePitch.default and voice.engine != 'standard':
                continue
            retval.append(GenerationConfig(
                voice=voice,
                rate=aug.rate,
                pitch=aug.pitch,
                keyword=None
            ))
        return retval


    def count_characters(self, config:GenerationConfig) -> int:
        assert config.voice.backend == self.name
        _, text = self._generate_msg(config)
        return len(text)

    def generate_filename(self, config:GenerationConfig) -> str:
        assert config.voice.backend == self.name
        return f'{self.name}_{config.voice.language_code}+{config.voice.name}+{config.keyword}+{config.rate}+{config.pitch}+{config.voice.hex_hash}.wav'

    def generate(self, config:GenerationConfig, out_dir:str) -> str:
        assert config.voice.backend == self.name

        out_path = f'{out_dir}/{self.generate_filename(config)}'
        if os.path.exists(out_path):
            import time
            time.sleep(.250)
            return out_path


        text_type, text = self._generate_msg(config)


        try:
            self.update_generate_timestamp()
            response = self._client.synthesize_speech(
                Engine=config.voice.engine,
                LanguageCode=config.voice.language_code,
                VoiceId=config.voice.name,
                SampleRate='16000',
                OutputFormat="pcm",
                Text=text,
                TextType=text_type
            )
            if 'AudioStream' in response:
                with contextlib.closing(response["AudioStream"]) as stream:
                    with wave.open(out_path, 'w') as wav:
                        wav.setframerate(16000) # pylint:disable=no-member
                        wav.setnchannels(1) # pylint:disable=no-member
                        wav.setsampwidth(2) # pylint:disable=no-member
                        wav.writeframesraw(stream.read()) # pylint:disable=no-member
            else:
                raise Exception(f'No audio stream in response: {response}')
        except Exception as e: # pylint:disable=redefined-outer-name
            try:
                os.remove(out_path)
            except:
                pass
            raise RuntimeError(f'{self.name} backend: Failed to generate: {out_path}, err: {e}')

        return out_path


    def _generate_msg(self, config:GenerationConfig) -> Tuple[str,str]:
        rate = config.rate
        pitch = config.pitch

        if config.rate in ('default', 'medium'):
            rate = None

        if pitch in ('default', 'medium'):
            pitch = None

        if rate or pitch:
            text_type = 'ssml'
            text = '<speak>'
            if rate:
                text += f'<prosody rate="{rate}">'
            if pitch:
                text += f'<prosody pitch="{pitch}">'

            text += config.keyword
            if pitch:
                text += '</prosody>'
            if rate:
                text += '</prosody>'

            text += '</speak>'

        else:
            text_type = 'text'
            text = config.keyword

        return text_type, text


@dataclass
class AwsVoice(Voice):
    engine:str=None

    def hashable_value(self) -> str:
        return self.name + self.language_code + self.backend + self.engine

    def __hash__(self):
        return hash((self.name, self.language_code, self.backend, self.engine))


