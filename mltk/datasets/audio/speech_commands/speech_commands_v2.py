"""Google Speech Commands v2
******************************************

https://www.tensorflow.org/datasets/catalog/speech_commands

This is a set of one-second .wav audio files, each containing a single spoken
English word. These words are from a small set of commands, and are spoken by a
variety of different speakers. The audio files are organized into folders based
on the word they contain, and this data set is designed to help train simple
machine learning models. This dataset is covered in more detail at 
`https://arxiv.org/abs/1804.03209 <https://arxiv.org/abs/1804.03209>`_.

It's licensed under the `Creative Commons BY 4.0
license <https://creativecommons.org/licenses/by/4.0/>`_. See the LICENSE
file in this folder for full details. Its original location was at
`http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz <http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz>`_.


History
---------

Version 0.01 of the data set was released on August 3rd 2017 and contained
64,727 audio files.

This is version 0.02 of the data set containing 105,829 audio files, released on
April 11th 2018.

Collection
---------------

The audio files were collected using crowdsourcing, see
`aiyprojects.withgoogle.com/open_speech_recording <https://github.com/petewarden/extract_loudest_section>`_
for some of the open source audio collection code we used (and please consider
contributing to enlarge this data set). The goal was to gather examples of
people speaking single-word commands, rather than conversational sentences, so
they were prompted for individual words over the course of a five minute
session. Twenty core command words were recorded, with most speakers saying each
of them five times. The core words are "Yes", "No", "Up", "Down", "Left",
"Right", "On", "Off", "Stop", "Go", "Zero", "One", "Two", "Three", "Four",
"Five", "Six", "Seven", "Eight", and "Nine". To help distinguish unrecognized
words, there are also ten auxiliary words, which most speakers only said once.
These include "Bed", "Bird", "Cat", "Dog", "Happy", "House", "Marvin", "Sheila",
"Tree", and "Wow".

Organization
----------------

The files are organized into folders, with each directory name labelling the
word that is spoken in all the contained audio files. No details were kept of
any of the participants age, gender, or location, and random ids were assigned
to each individual. These ids are stable though, and encoded in each file name
as the first part before the underscore. If a participant contributed multiple
utterances of the same word, these are distinguished by the number at the end of
the file name. For example, the file path `happy/3cfc6b3a_nohash_2.wav`
indicates that the word spoken was "happy", the speaker's id was "3cfc6b3a", and
this is the third utterance of that word by this speaker in the data set. The
'nohash' section is to ensure that all the utterances by a single speaker are
sorted into the same training partition, to keep very similar repetitions from
giving unrealistically optimistic evaluation scores.
 

Processing
---------------

The original audio files were collected in uncontrolled locations by people
around the world. We requested that they do the recording in a closed room for
privacy reasons, but didn't stipulate any quality requirements. This was by
design, since we wanted examples of the sort of speech data that we're likely to
encounter in consumer and robotics applications, where we don't have much
control over the recording equipment or environment. The data was captured in a
variety of formats, for example Ogg Vorbis encoding for the web app, and then
converted to a 16-bit little-endian PCM-encoded WAVE file at a 16000 sample
rate. The audio was then trimmed to a one second length to align most
utterances, using the
`extract_loudest_section <https://github.com/petewarden/extract_loudest_section>`_
tool. The audio files were then screened for silence or incorrect words, and
arranged into folders by label.

Background Noise
----------------

To help train networks to cope with noisy environments, it can be helpful to mix
in realistic background audio. The `_background_noise_` folder contains a set of
longer audio clips that are either recordings or mathematical simulations of
noise. For more details, see the `_background_noise_/README.md`.

Citations
---------

If you use the Speech Commands dataset in your work, please cite it as:

::

  @article{speechcommandsv2,
    author = {{Warden}, P.},
      title = "{Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition}",
    journal = {ArXiv e-prints},
  archivePrefix = "arXiv",
    eprint = {1804.03209},
  primaryClass = "cs.CL",
  keywords = {Computer Science - Computation and Language, Computer Science - Human-Computer Interaction},
      year = 2018,
      month = apr,
      url = {https://arxiv.org/abs/1804.03209},
  }


Credits
---------

Massive thanks are due to everyone who donated recordings to this data set, I'm
very grateful. I also couldn't have put this together without the help and
support of Billy Rutledge, Rajat Monga, Raziel Alvarez, Brad Krueger, Barbara
Petit, Gursheesh Kour, and all the AIY and TensorFlow teams.

Pete Warden, petewarden@google.com

"""
import os
import re
import math
from typing import List, Tuple
from mltk.utils.archive_downloader import download_verify_extract
from mltk.core.utils import get_mltk_logger


DOWNLOAD_URL = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
VERIFY_SHA1 = '4264eb9753e38eef2ec1d15dfac8441f09751ca9'



def load_data() -> str:
    """Download and extract the Google Speech commands dataset v2, 
    and return the directory path to the extracted dataset
    """

    path = download_verify_extract(
        url=DOWNLOAD_URL,
        dest_subdir='datasets/speech_commands/v2',
        file_hash=VERIFY_SHA1,
        show_progress=True
    )
    return path


def list_valid_filenames_in_directory(
    base_directory:str, 
    search_class:str, 
    white_list_formats:List[str], 
    split:float, 
    follow_links:bool, 
    shuffle_index_directory:str
) -> Tuple[str, List[str]]:
    """Return a list of valid file names for the given class
  
  
    Per the dataset README.md:

    We want to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that testing
    samples will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.

    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_nohash_' in a filename is
    ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    'bobby_nohash_1.wav' are always in the same set, for example.
    """
    assert shuffle_index_directory is None, 'Shuffling the index is not supported by this dataset'

    file_list = []
    index_path = f'{base_directory}/.index/{search_class}.txt'


    # If the index file exists, then read it
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            for line in f:
                file_list.append(line.strip())

    else:
        get_mltk_logger().info(f'Generating index: {index_path} ...')
        # Else find all files for the given class in the search directory
        for root, _, files in os.walk(base_directory, followlinks=follow_links):
            if os.path.basename(root) != search_class:
                continue
            
            for fname in files:
                if not fname.lower().endswith(white_list_formats):
                    continue
                abs_path = os.path.join(root, fname)
                rel_path = os.path.relpath(abs_path, base_directory)
                file_list.append(rel_path.replace('\\', '/'))


            # Sort the filenames alphabetically
            file_list = sorted(file_list)

            # Write the file list file
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            with open(index_path, 'w') as f:
                for p in file_list:
                    f.write(p + '\n')


    if split:
        get_file_hash = lambda x: re.sub(r'_nohash_.*$', '', os.path.basename(x)) 
        num_files = len(file_list)
        if split[0] == 0:
            start = 0
            stop = math.ceil(split[1] * num_files)

            # We want to ensure the same person isn't in both subsets
            # So, ensure that the split point does NOT
            # split with file names with the same hash
            # recall: same hash = same person saying word

            # Get the hash of the other subset
            other_subset_hash = get_file_hash(file_list[stop])
            # Keep moving the 'stop' index back while
            # it's index matches the otherside
            while stop > 0 and get_file_hash(file_list[stop-1]) == other_subset_hash:
              stop -= 1

        else:
            start = math.ceil(split[0] * num_files)
            # Get the hash of the this subset
            this_subset_hash = get_file_hash(file_list[start])
            # Keep moving the 'start' index back while
            # it's index matches this side's
            while start > 0 and get_file_hash(file_list[start-1]) == this_subset_hash:
              start -= 1

            stop = num_files

        filenames = file_list[start:stop] 
    
    else:
        filenames = file_list

    return search_class, filenames

