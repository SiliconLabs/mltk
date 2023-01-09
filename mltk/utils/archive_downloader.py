"""Utilities for downloading and extracting archives

See the source code on Github: `mltk/utils/archive_downloader.py <https://github.com/siliconlabs/mltk/blob/master/mltk/utils/archive_downloader.py>`_
"""



import sys
import os
import hashlib
import json
import shutil
import logging
import urllib.request
from urllib.parse import urlsplit

try:
    from tqdm import tqdm
    have_tqdm = True
except:
    have_tqdm = False


from .archive import extract_archive
from .logger import get_logger
from .path import create_user_dir, fullpath
from .python import prepend_exception_msg


MLTK_CHUNK_DELIMITER = '?mltk_chunk_count='


def download_verify_extract(
    url: str,
    dest_dir:str=None,
    dest_subdir:str=None,
    download_dir:str=None,
    archive_fname:str=None,
    show_progress:bool=False,
    file_hash:str=None,
    file_hash_algorithm:str='auto',
    logger:logging.Logger=None,
    extract_nested:bool=False,
    remove_root_dir:bool=False,
    clean_dest_dir:bool=True,
    update_onchange_only:bool=True,
    download_details_fname:str=None,
    extract:bool=True
) -> str:
    """Download an archive, verify its hash, and extract

    Args:
        url: Download URL
        dest_dir: Directory to extract archive into
                  If omitted, defaults to MLTK_CACHE_DIR/<dest_subdir>/ OR
                  ~/.mltk/<dest_subdir>/
        dest_subdir: Destination sub-directory, if omitted default to archive path's basename
                     This is only used if dest_dir is omitted
        download_dir: Directory to download archive to
                 If omitted, defaults to MLTK_CACHE_DIR/downloads/<archive_fname> OR
                 ~/.mltk/downloads/<archive_fname>
        archive_fname: Name of downloaded archive file, if omitted default to URL filename
        show_progress: Show a download progressbar
        file_hash: md5, sha1, sha256 hash of file
        file_hash_algorithm: File hashing algorithm, if auto then determine automatically
        extract_nested: If the archive has a sub archive, then extract that as well
        remove_root_dir: If the archive has a root directory, then remove it from the extracted path
        clean_dest_dir: Remove the destination directory BEFORE extracting
        update_onchange_only: Only download and extract if given url hasn't been previously downloaded and extracted, otherwise return immediately
        download_details_fname: If update_onchange_only=True then a download details .json file is generated.
                                This argument specifies the name of that file. If omitted, then the filename is <archive filename>-mltk.json
        extract: If false, then do NOT extract the downloaded file. In this case, return the path to the downloaded file

    Returns:
        Path to extracted directory OR path to downloaded archive is extract=False
    """
    logger = logger or get_logger()

    if not archive_fname:
        archive_fname = os.path.basename(urlsplit(url).path)
        if not archive_fname:
            raise ValueError('Failed to determine archive filename or given URL')

    if not download_dir:
        download_dir = create_user_dir(suffix='downloads')
    else:
        download_dir = create_user_dir(base_dir=download_dir)

    archive_path = f'{download_dir}/{archive_fname}'
    download_details_fname = download_details_fname or f'{archive_fname}-mltk.json'

    if not extract:
        retval = archive_path
        downloads_details_path = f'{download_dir}/{download_details_fname}'
    elif not dest_dir:
        subdir = dest_subdir or os.path.splitext(archive_fname)[0]
        retval = create_user_dir(suffix=subdir)
        downloads_details_path = f'{retval}/{download_details_fname}'
    else:
        retval = create_user_dir(base_dir=dest_dir)
        downloads_details_path = f'{retval}/{download_details_fname}'


    download_details = dict(
        url=url,
        retval=retval,
        archive_path=archive_path,
        file_hash=file_hash,
        file_hash_algorithm=file_hash_algorithm,
        remove_root_dir=remove_root_dir
    )


    if update_onchange_only:
        if _check_if_up_to_date(
            details_path=downloads_details_path,
            details=download_details
        ):
            logger.debug(f'Up-to-date: {url} -> {retval}')
            return retval

    for i in range(2):
        # Download the archive or use the cached version in the download_dir
        download_url(
            url,
            dst_path=archive_path,
            show_progress=show_progress,
            logger=logger
        )

        try:
            if file_hash and not verify_file_hash(
                    file_path=archive_path,
                    file_hash=file_hash,
                    file_hash_algorithm=file_hash_algorithm
                ):
                raise Exception('File hash invalid')

            # The downloaded version was valid, so continue to extraction
            break
        except Exception as e:
            # Remove the cached version
            try:
                os.remove(archive_path)
            except:
                pass
            # If this was the first attempt,
            # Then continue to the beginning and try one more time
            # by re-downloading the file instead of using the cache downloaded archive
            if i == 0:
                logger.debug(f'Download failed: {e}, retrying')
                continue

            # Otherwise just through the exception
            raise e

    if extract:
        logger.warning(f"Extracting: {archive_path}\nto: {retval}\n(This may take awhile, please be patient ...)")
        extract_archive(
            archive_path=archive_path,
            dest_dir=retval,
            extract_nested=extract_nested,
            clean_dest_dir=clean_dest_dir,
            remove_root_dir=remove_root_dir
        )


    if update_onchange_only:
        with open(downloads_details_path, 'w') as f:
            json.dump(download_details, f, indent=3)

    return retval




def verify_extract(
    archive_path: str,
    dest_dir:str=None,
    dest_subdir:str=None,
    show_progress:bool=False,
    file_hash:str=None,
    file_hash_algorithm:str='auto',
    logger:logging.Logger=None,
    extract_nested:bool=False,
    remove_root_dir:bool=False,
    clean_dest_dir:bool=True,
    update_onchange_only:bool=True,
    extract_details_fname:str=None
) -> str:
    """Verify the archive hash and extract

    Args:
        archive_path: File path to archive
        dest_dir: Directory to extract archive into
                  If omitted, defaults to MLTK_CACHE_DIR/<dest_subdir>/ OR
                  ~/.mltk/<dest_subdir>/
        dest_subdir: Destination sub-directory, if omitted default to archive path's basename
                     This is only used if dest_dir is omitted
        show_progress: Show a download progressbar
        file_hash: md5, sha1, sha256 hash of file
        file_hash_algorithm: File hashing algorithm, if auto then determine automatically
        extract_nested: If the archive has a sub archive, then extract that as well
        remove_root_dir: If the archive has a root directory, then remove it from the extracted path
        clean_dest_dir: Remove the destination directory BEFORE extracting
        update_onchange_only: Only download and extract if given url hasn't been previously downloaded and extracted, otherwise return immediately
        extract_details_fname: If update_onchange_only=True then a details .json file is generated.
                                This argument specifies the name of that file. If omitted, then the filename is <archive filename>-mltk.json

    Returns:
        Path to extracted directory
    """
    logger = logger or get_logger()

    if not os.path.exists(archive_path):
        raise FileNotFoundError(f'Archive not found at {archive_path}')

    archive_fname = os.path.basename(archive_path)

    if not dest_dir:
        subdir = dest_subdir or os.path.splitext(archive_fname)[0]
        dest_dir = create_user_dir(suffix=subdir)
    else:
        dest_dir = create_user_dir(base_dir=dest_dir)


    extract_details_fname = extract_details_fname or f'{archive_fname}-mltk.json'
    extract_details_path = f'{dest_dir}/{extract_details_fname}'
    extract_details = dict(
        archive_fname=archive_fname,
        dest_dir=dest_dir,
        archive_path=archive_path,
        file_hash=file_hash,
        file_hash_algorithm=file_hash_algorithm,
        timestamp = os.path.getmtime(archive_path),
        remove_root_dir=remove_root_dir
    )


    if update_onchange_only:
        if _check_if_up_to_date(
            details_path=extract_details_path,
            details=extract_details,
        ):
            logger.debug(f'Up-to-date: {archive_path} -> {dest_dir}')
            return dest_dir


    if file_hash and not verify_file_hash(
            file_path=archive_path,
            file_hash=file_hash,
            file_hash_algorithm=file_hash_algorithm
        ):
        raise Exception('File hash invalid')



    logger.warning(f"Extracting: {archive_path}\nto: {dest_dir}\n(This may take awhile, please be patient ...)")
    extract_archive(
        archive_path=archive_path,
        dest_dir=dest_dir,
        extract_nested=extract_nested,
        clean_dest_dir=clean_dest_dir,
        remove_root_dir=remove_root_dir
    )

    if update_onchange_only:
        with open(extract_details_path, 'w') as f:
            json.dump(extract_details, f, indent=3)

    return dest_dir


def download_url(
    url:str,
    dst_path:str,
    show_progress=False,
    logger=None
) -> str:
    """Downloads the tarball or zip file from url into dst_path.
    Args:
      url: The URL of a tarball or zip file.
      dst_path: The path where the file is download
      show_progress: Show a progress bar while downloading

    If the file at ``dst_path`` is already found,
    then just return the local version without downloading
    """
    logger = logger or get_logger()

    dst_path = fullpath(dst_path)

    # If the file has already been downloaded
    # then just return that
    if os.path.exists(dst_path):
        logger.debug(f'Using cached: {url}\nat: {dst_path}')
        return

    if MLTK_CHUNK_DELIMITER in url:
        _download_chunks(
            url,
            dst_path=dst_path,
            logger=logger,
            show_progress=show_progress
        )
        return dst_path




    tmp_filepath = dst_path + '.tmp'
    try:
        os.remove(tmp_filepath)
    except:
        pass

    os.makedirs(os.path.dirname(tmp_filepath), exist_ok=True)

    logger.warning(f'Downloading {url}\nto {dst_path}\n(This may take awhile, please be patient ...)')
    try:
        if show_progress and have_tqdm:
            with _ProgressBar(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=url, leave=False) as t:
                tmp_filepath, _ = urllib.request.urlretrieve(url, tmp_filepath, t.update_chunk)
        else:
            tmp_filepath, _ = urllib.request.urlretrieve(url, tmp_filepath)

        shutil.move(tmp_filepath, dst_path)

    except Exception as e:
        try:
            os.remove(tmp_filepath)
        except:
            pass
        prepend_exception_msg(e, f'Failed to download: {url}')
        raise

    return dst_path


def verify_file_hash(
    file_path:str,
    file_hash:str,
    file_hash_algorithm:str
):
    """Return True if the calculated hash of the file matches the given hash, false else"""

    md5_hasher = hashlib.md5()
    sha1_hasher = hashlib.sha1()
    sha256_hasher = hashlib.sha256()

    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hasher.update(chunk)
            sha1_hasher.update(chunk)
            sha256_hasher.update(chunk)

    calc_md5_hash = md5_hasher.hexdigest().lower()
    calc_sha1_hash = sha1_hasher.hexdigest().lower()
    calc_sha256_hash = sha256_hasher.hexdigest().lower()

    file_hash = file_hash.lower()

    if file_hash_algorithm in ('auto', 'md5') and calc_md5_hash == file_hash:
        return True
    if file_hash_algorithm in ('auto', 'sha1') and calc_sha1_hash == file_hash:
        return True
    if file_hash_algorithm in ('auto', 'sha256') and calc_sha256_hash == file_hash:
        return True

    return False


def verify_sha1(file_path, expected_sha1):
    with open(file_path, 'rb') as f:
        hasher = hashlib.sha1()
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    calc_hash = hasher.hexdigest().lower()

    if callable(expected_sha1):
        expected_sha1(calc_hash)
        return

    expected_sha1 = expected_sha1.lower()
    if calc_hash != expected_sha1:
        raise Exception(f'Calculated hash ({calc_hash}) does not match expected hash ({expected_sha1})')


def verify_sha256(file_path, expected_sha256):
    with open(file_path, 'rb') as f:
        hasher = hashlib.sha256()
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    calc_hash = hasher.hexdigest().lower()

    if callable(expected_sha256):
        expected_sha256(calc_hash)
        return

    expected_sha256 = expected_sha256.lower()
    if calc_hash != expected_sha256:
        raise Exception(f'Calculated hash ({calc_hash}) does not match expected hash ({expected_sha256})')


def _check_if_up_to_date(
    details_path:str,
    details:dict
):
    try:
        with open(details_path, 'r') as f:
            loaded_details = json.load(f)
        if loaded_details == details:
            return os.path.exists(details['retval'])
    except:
        pass

    return False


def _download_chunks(
    url:str,
    dst_path:str,
    show_progress=False,
    logger=None
):
    delimiter_index = url.find(MLTK_CHUNK_DELIMITER)
    chunk_count = int(url[delimiter_index + len(MLTK_CHUNK_DELIMITER):])
    url = url[:delimiter_index]

    tmp_filepath = dst_path + '.tmp'
    try:
        os.remove(tmp_filepath)
    except:
        pass


    logger.warning(f'Downloading {url}\nto {dst_path}\n(This may take awhile, please be patient ...)')
    chunk_paths = []

    try:
        if show_progress and have_tqdm:
            for chunkno in range(chunk_count):
                chunk_url = f'{url}.chunk{chunkno}.bin'
                chunk_path = f'{dst_path}.chunk{chunkno}.bin'
                chunk_paths.append(chunk_path)
                with _ProgressBar(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=chunk_url, leave=False) as t:
                    t.set_chunkno(chunkno, chunk_count)
                    urllib.request.urlretrieve(chunk_url, chunk_path, t.update_chunk)
        else:
            for chunkno in range(chunk_count):
                chunk_url = f'{url}.chunk{chunkno}.bin'
                chunk_path = f'{dst_path}.chunk{chunkno}.bin'
                chunk_paths.append(chunk_path)
                urllib.request.urlretrieve(chunk_url, chunk_path)

        with open(tmp_filepath, 'wb') as dst:
            for chunk_path in chunk_paths:
                with open(chunk_path, 'rb') as src:
                    shutil.copyfileobj(src, dst)

        shutil.move(tmp_filepath, dst_path)
    except Exception as e:
        prepend_exception_msg(e, f'Failed to download chunks: {url}')
        raise
    finally:
        for chunk_path in chunk_paths:
            try:
                os.remove(chunk_path)
            except:
                pass




if have_tqdm:
    class _ProgressBar(tqdm):
        def __init__(self, *args, **kwargs):
            tqdm.__init__(self, *args, file=sys.stdout, **kwargs)

        def update_chunk(self, b=1, bsize=1, tsize=None):
            """
            b  : int, optional
                Number of blocks transferred so far [default: 1].
            bsize  : int, optional
                Size of each block (in tqdm units) [default: 1].
            tsize  : int, optional
                Total size (in tqdm units). If [default: None] remains unchanged.
            """
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)  # will also set self.n = b * bsize

        def set_chunkno(self, chunkno:int, total:int):
            self.set_postfix_str(f'Chunk {chunkno+1} of {total}')
