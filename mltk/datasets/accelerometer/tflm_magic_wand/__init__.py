"""Tensorflow-Lite Micro Magic Wand
***************************************

https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/magic_wand



"""

import os
import sys
from typing import Tuple, List
import tensorflow as tf
from mltk.utils.archive_downloader import download_verify_extract

if not __package__:     
    CURDIR = os.path.dirname(os.path.abspath(__file__))             
    sys.path.insert(0, os.path.dirname(CURDIR))
    __package__ = os.path.basename(CURDIR)# pylint: disable=redefined-builtin



from .data_prepare import data_prepare
from .data_split_person import data_split_person
from .data_load import DataLoader


DOWNLOAD_URL = 'http://download.tensorflow.org/models/tflite/magic_wand/data.tar.gz'
VERIFY_SHA1 = '5F130F7A65DB62E17E7DE62D635AB7AE0F929047'



def prepare_data() -> str:
    """Download and prepare the dataset, then return the path to the dataset
    """

    dataset_dir = download_verify_extract(
        url=DOWNLOAD_URL,
        dest_subdir='datasets/accelerator/tflm_magic_wand',
        file_hash=VERIFY_SHA1,
        show_progress=True
    )

    processing_complete_path = f'{dataset_dir}/processing_complete.txt'
    if not os.path.exists(processing_complete_path):
        try:
            saved_cwd_dir = os.getcwd()
            os.chdir(dataset_dir)
            data_prepare()
            data_split_person()

            with open(processing_complete_path, 'w'):
                pass

        finally:
            os.chdir(saved_cwd_dir)

    return dataset_dir


def load_data(seq_length=128, person=True) -> DataLoader:
    """Download and prepare the dataset, then return the path to the dataset

    as a tuple:
    (train, validation, test)
    """
    dataset_dir = prepare_data()
    
    subdir = 'person_split' if person else 'data' 
    data_loader = DataLoader(
        f'{dataset_dir}/{subdir}/train',
        f'{dataset_dir}/{subdir}/valid',
        f'{dataset_dir}/{subdir}/test',
        seq_length=seq_length
    )

    return data_loader


if __name__ == '__main__':
    load_data()