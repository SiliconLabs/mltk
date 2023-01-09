from dataclasses import dataclass
from typing import List,Tuple, Union
import struct


@dataclass
class AdpcmState:
    predicted_sample:int = 0
    step_index:int = 0



def decode_mono_data(data:bytes, state:AdpcmState=None) -> Tuple[List[int], AdpcmState]:
    retval = []
    for c in data:
        state = decode_sample(c & 0x0F, state)
        retval.append(state.predicted_sample)
        state = decode_sample(c >> 4, state)
        retval.append(state.predicted_sample)

    return retval, state


def encode_mono_data(data:Union[List[int],bytes], state:AdpcmState=None) -> Tuple[bytes, AdpcmState]:
    assert len(data) % 2 == 0, 'Data must be a multip,le of 2'

    if isinstance(data, (bytes,bytearray)):
        data = bytes_to_pcm(data)

    encoded = bytearray()

    for i in range(0, len(data), 2):
        nibble1, state = encode_sample(data[i], state)
        nibble2, state = encode_sample(data[i+1], state)
        encoded.append(nibble2 << 4 | nibble1)

    return encoded, state




def decode_sample(sample:int, state: AdpcmState=None) -> AdpcmState:
    state = state or AdpcmState()

    step = _STEPS[state.step_index]

    diff = step >> 3
    if sample & 1: 
        diff += step >> 2
    if sample & 2: 
        diff += step >> 1
    if sample & 4: 
        diff += step
    if sample & 8: 
        diff = -diff
    
    state.predicted_sample = _clamp(state.predicted_sample + diff, -32768, 32767)
    state.step_index = _clamp(state.step_index + _STEP_INDICES[sample & 7], 0, 88)

    return state


def encode_sample(sample:int, state: AdpcmState=None) -> Tuple[int, AdpcmState]:
    state = state or AdpcmState()
    step = _STEPS[state.step_index]

    sample_diff = sample - state.predicted_sample
    encoded_sample = 8 if (sample_diff < 0) else 0

    if encoded_sample:
        sample_diff = -sample_diff
    
    diff = step >> 3
    if sample_diff >= step:
        encoded_sample |= 4
        sample_diff -= step
        diff += step

    step >>= 1
    if sample_diff >= step:
        encoded_sample |= 2
        sample_diff -= step
        diff += step

    step >>= 1
    if sample_diff >= step:
        encoded_sample |= 1
        diff += step

    if encoded_sample & 8:
        diff = -diff
    
    state.predicted_sample = _clamp(state.predicted_sample + diff, -32768, 32767)
    state.step_index = _clamp(state.step_index + _STEP_INDICES[encoded_sample & 7], 0, 88)

    return encoded_sample, state


def bytes_to_pcm(d:bytes) -> Tuple[int]:
    """Convert a byte string to a list of signed, 16-bit integers (i.e. PCM)"""
    return tuple(struct.unpack('<h', x)[0] for x in d)

def pcm_to_bytes(d:Tuple[int]) -> bytes:
    """Convert a list of signed, 16-bit integers (i.e. PCM) to byte string"""
    retval = bytearray()
    for x in d:
        retval.extend(struct.pack('<h', x))
    return retval



def _clamp(value, lower, upper):
    return max(min(value, upper), lower)



_STEP_INDICES = (-1, -1, -1, -1, 2, 4, 6, 8)
_STEPS = (
    7, 8, 9, 10, 11, 12, 13, 14,
    16, 17, 19, 21, 23, 25, 28, 31,
    34, 37, 41, 45, 50, 55, 60, 66,
    73, 80, 88, 97, 107, 118, 130, 143,
    157, 173, 190, 209, 230, 253, 279, 307,
    337, 371, 408, 449, 494, 544, 598, 658,
    724, 796, 876, 963, 1060, 1166, 1282, 1411,
    1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024,
    3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484,
    7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
    15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794,
    32767
)