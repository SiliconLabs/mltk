"""Generic Background Noise
****************************
"""
from typing import List, Union, Tuple
import os
from urllib.parse import urlparse
import logging
from mltk.utils.python import as_list
from mltk.utils.archive_downloader import download_url, download_verify_extract
from mltk.core.preprocess.utils import audio as audio_utils


def download(
    dest_dir:str,
    urls:Union[str,List[str]],
    sample_rate_hertz:int=16000,
    logger:logging.Logger=None
) -> List[str]:
    os.makedirs(dest_dir, exist_ok=True)
    urls = as_list(urls)
    retval = []

    for url in urls:
        p = urlparse(url).path
        fn = os.path.basename(p)
        dst_path = f'{dest_dir}/{fn}'
        retval.append(dst_path)

        if not os.path.exists(dst_path):
            download_url(
                url=url,
                dst_path=dst_path,
                logger=logger
            )
            sample, original_sample_rate = audio_utils.read_audio_file(
                dst_path,
                return_sample_rate=True,
                return_numpy=True
            )

            if original_sample_rate != sample_rate_hertz:
                sample = audio_utils.resample(
                    sample,
                    orig_sr=original_sample_rate,
                    target_sr=sample_rate_hertz
                )
                audio_utils.write_audio_file(
                    dst_path,
                    sample,
                    sample_rate=sample_rate_hertz
                )

    return retval


def download_and_extract(
    dest_dir:str,
    urls:Union[str, Tuple[str,str], List[str], List[Tuple[str,str]]],
    logger:logging.Logger=None,
    clean_dest_dir=False
) -> List[str]:
    os.makedirs(dest_dir, exist_ok=True)
    if isinstance(urls, tuple):
        urls = [urls]
    else:
        urls = as_list(urls)
    retval = []

    for url_and_hash in urls:
        if isinstance(url_and_hash, str):
            url = url_and_hash
            hash = None
        else:
            url = url_and_hash[0]
            hash = url_and_hash[1]

        sample_dir = download_verify_extract(
            url=url,
            dest_dir=dest_dir,
            file_hash=hash,
            show_progress=False,
            remove_root_dir=False,
            clean_dest_dir=clean_dest_dir,
            logger=logger
        )
        clean_dest_dir = False
        retval.append(sample_dir)

    return retval

