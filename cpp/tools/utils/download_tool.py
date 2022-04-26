"""Downloads a tool archive to the specified directory"""

from typing import Tuple
import sys
import traceback
import logging

from mltk.utils.logger import get_logger, make_filelike
from mltk.utils.path import remove_directory, create_tempdir


def download_tool(
    url:str, 
    dest_dir:str, 
    name:str, 
    extracted_dir:str=None, 
    show_progress=True, 
    file_hash:str=None,
    extract_nested=False, 
    clean_dest_dir=True,
    logger: logging.Logger = None,
    log_file:str=None,
    download_details_fname:str=None,
    override_stdout_with_logger=False
) -> Tuple[bool, str]:
    """Download and extract a tool."""

    logger = logger or get_logger('mltk', level='DEBUG', log_file=log_file, log_file_mode='w')

    make_filelike(logger)

    if override_stdout_with_logger:
        saved_stdout = sys.stdout
        saved_stderr = sys.stderr
        sys.stdout = logger
        sys.stderr = logger

    # We need to import the archive extraction package
    # AFTER overridding it in sys.stdout
    from mltk.utils.archive_downloader import download_verify_extract

    
    # Some archives have a directory in its root
    # The extracted_dir accounts for the root directory
    retval = extracted_dir or dest_dir

    if clean_dest_dir and extracted_dir:
        def _clean_dest_dir():
            remove_directory(extracted_dir)
        clean_dest_dir = _clean_dest_dir

    retcode = -1 

    try:
        download_verify_extract(
            url=url, 
            dest_dir=dest_dir, 
            extract_nested=extract_nested,
            file_hash=file_hash,
            show_progress=show_progress, 
            logger=logger,
            clean_dest_dir=clean_dest_dir,
            update_onchange_only=True,
            download_details_fname=download_details_fname
        )
        retcode = 0
    except Exception as e:
        logger.error(f'Error while downloading {name}, err: {e}', exc_info=e)
        error_msg = f'\n\nError while downloading tool, err: {e}'
        retval = create_tempdir('temp') + f'/{name}_install_error.txt'
        with open(retval, 'w') as f:
            f.write(error_msg)
    finally:
        if override_stdout_with_logger:
            sys.stderr = saved_stderr
            sys.stdout = saved_stdout

    return retcode, retval