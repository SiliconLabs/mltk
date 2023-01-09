import wave
import os 
import sys
import struct
import pathlib

if __package__ is None:                  
    DIR = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(DIR.parent))
    __package__ = DIR.name



from mltk.utils.bin2header import bin2header

curdir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
test_mp3_path = f'{curdir}/test.mp3'


bin2header(
    input=test_mp3_path,
    output_path=f'{curdir}/test_mp3_data.h',
    var_name='MP3_DATA',
    length_var_name='MP3_DATA_LENGTH',
    dtype='const uint8_t',
    prepend_lines=[ 
        '#pragma once',
        '#include <stdint.h>',
    ]
)