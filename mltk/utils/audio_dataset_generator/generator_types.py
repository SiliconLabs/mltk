from __future__ import annotations
import abc
import copy
from typing import List, NamedTuple
import time
import threading
import logging
from enum import Enum
import hashlib
from dataclasses import dataclass


logger = logging.getLogger('AudioDatasetGenerator')


class Keyword(NamedTuple):
    """Keyword to generate

    """
    value:str
    """The base keyword

    .. note:: If the base keyword starts with an underscore,
        then it is NOT included in the list of keywords to generate (see :py:func:`~as_list`).
        In this case, the :py:attr:`~aliases` must be provided.
    """
    aliases:List[str]=None
    """Additional aliases for the base keyword"""
    max_count:int=None
    """The maximum number of samples to generate for this keyword
    This is only used if the ``truncate`` argument of :py:func:`~AudioDatasetGenerator.list_configurations` is ``true```
    """

    def as_list(self) -> List[str]:
        """Return the base keyword and all its aliases as a list of strings

        .. note:: If the base keyword (e.g. :py:attr:`~value`) starts with an underscore, then it is omitted from the list
        """
        retval = []
        if not self.value.startswith('_'):
            retval.append(self.value)
        if self.aliases:
            for alias in self.aliases: # pylint:disable=not-an-iterable
                retval.append(alias)
        return retval

    def __str__(self) -> str:
        return self.value


class Augmentation(NamedTuple):
    """Augmentations to apply to the audio sample"""

    pitch:VoicePitch=None
    """The pitch of the voice"""
    rate:VoiceRate=None
    """The speaking rate of the voice"""

    def __str__(self) -> str:
        return f'Pitch:{self.pitch.value} Rate:{self.rate.value}'



class VoicePitch(str,Enum):
    """The "pitch" of the voice used in the audio sample"""
    low = 'low'
    medium = 'medium'
    high = 'high'
    default = 'medium'


class VoiceRate(str,Enum):
    """The speaking rate of the voice used in the audio sample"""
    xslow = 'x-slow'
    medium = 'medium'
    xfast = 'x-fast'
    default = 'medium'



@dataclass
class Voice:
    """The backend voice used to generate a keyword"""
    name:str
    """The name of the voice as specified by the backend"""
    language_code:str
    """The language code of the voice as specified by the backend"""
    backend:str
    """The name of the voice's backend"""

    def hashable_value(self) -> str:
        """The value used to generate a unique "hash" for the voice"""
        return self.name + self.language_code + self.backend

    @property
    def hex_hash(self) -> str:
        """A unique hash for the voice
        This may be used to group samples so that the same voice does not
        appear in the "training" and "validation" subsets
        """
        if not hasattr(self, '_hex_hash'):
            hasher = hashlib.sha1()
            hasher.update(self.hashable_value().encode('utf-8'))
            setattr(self, '_hex_hash', hasher.hexdigest()[:8])
        return getattr(self, '_hex_hash')

    def __hash__(self):
        return hash((self.name, self.language_code, self.backend))


@dataclass
class GenerationConfig:
    """Audio sample generation configuration"""
    voice:Voice
    """The backend voice"""
    rate:VoiceRate
    """The speaking rate"""
    pitch:VoicePitch
    """The voice pitch"""
    keyword:str=None
    """The keyword text (this is either the base keyword or keyword alias)"""
    keyword_group:str=None
    """The base keyword"""

    def copy(self) -> GenerationConfig:
        """Return a deep copy of the configuration"""
        return copy.deepcopy(self)




class BackendBase(abc.ABC):
    """Base class for a cloud backend"""
    def __init__(self, transactions_per_second:float):
        super().__init__()
        self._generate_timestamp:float = 0.0
        self._min_seconds_per_transaction = 1/(transactions_per_second * 0.85)
        self._lock = threading.Lock()


    @property
    def is_rate_limited(self) -> bool:
        with self._lock:
            now = time.time()
            elapsed = now - self._generate_timestamp
            return elapsed < self._min_seconds_per_transaction

    def update_generate_timestamp(self):
        with self._lock:
            self._generate_timestamp = time.time()



    @property
    @abc.abstractproperty
    def name(self) -> str:
        ...

    @abc.abstractmethod
    def load(self, install_python_package=False, **kwargs):
        ...


    @abc.abstractmethod
    def list_languages(self) -> List[str]:
        ...

    @abc.abstractmethod
    def list_voices(self, language_code:str=None) -> List[Voice]:
        ...

    @abc.abstractmethod
    def list_configurations(
        self,
        augmentations:List[Augmentation],
        voice:Voice,
    ) -> List[GenerationConfig]:
        ...


    @abc.abstractmethod
    def count_characters(self, config:GenerationConfig) -> int:
        ...

    @abc.abstractmethod
    def generate(self, config:GenerationConfig, out_dir:str) -> str:
        ...

    @abc.abstractmethod
    def generate_filename(self, config:GenerationConfig) -> str:
        ...

