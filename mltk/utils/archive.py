"""Utilities for extracting archives

See the source code on Github: `mltk/utils/archive.py <https://github.com/siliconlabs/mltk/blob/master/mltk/utils/archive.py>`_
"""
import os
import re
import tarfile
import gzip
import struct
import shutil
from typing import Callable, Union
from patoolib.programs import tar # pylint: disable=unused-import
import patoolib

from .python import append_exception_msg, prepend_exception_msg
# We use a custom zipfile class which allows
# for extracting large zipfiles on Windows
from .zipfile_win32 import ZipFile
from . import path



def extract_archive(
    archive_path:str,
    dest_dir:str,
    extract_nested:bool=False,
    remove_root_dir:bool=False,
    clean_dest_dir:Union[bool,Callable]=True
):
    """Extract the given archive file to the specified directory

    Args:
        archive_path: Path to archive file
        dest_dir: Path to directory where archive will be extracted
        extract_nested: If true and the give archive contains nested archive, then extract those as well
        remove_root_dir: If the archive has a root directory, then remove it from the extracted path
        clean_dest_dir: Clean the destination directory before extracting
    """
    if clean_dest_dir:
        if callable(clean_dest_dir):
            clean_dest_dir()
        else:
            path.remove_directory(dest_dir)

    try:
        if extract_nested or remove_root_dir:
            _extractnested_archive(
                archive_path,
                dest_dir,
                extract_nested=extract_nested,
                remove_root_dir=remove_root_dir
            )

        elif archive_path.endswith('.zip'):
            _extractall_zipfile(archive_path, dest_dir)

        elif archive_path.endswith('.tar.gz'):
            _extractall_tarfile(archive_path, dest_dir)

        elif archive_path.endswith('.gz'):
            _extractall_gzfile(archive_path, dest_dir)

        else:
            _extractall_patool(archive_path, dest_dir)
    except Exception as e:
        prepend_exception_msg(e, f'Failed to extract {archive_path} to {dest_dir}')
        raise



def gzip_file(src_path : str, dst_path: str=None) -> str:
    """GZip file and return path to gzip archive

    Args:
        src_path: Path to local file to gzip
        dst_path: Optional path to destination gzip file. If omitted then use src_path + .gz

    Return:
        Path to generated .gz file
    """
    if not dst_path:
        dst_path = src_path + '.gz'

    with open(src_path, 'rb') as src:
        with gzip.open(dst_path, 'wb') as dst:
            shutil.copyfileobj(src, dst)

    return dst_path



def gzip_directory_files(
    src_dir:str,
    dst_archive:str = None,
    regex:Union[str,re.Pattern,Callable[[str],bool]]=None,
) -> str:
    """Recursively gzip all files in given directory.
    The generated .tar.gz contains the same directory structure as the src_dir.

    Args:
        src_dir: Path to directory to generated .tar.gz archive
        dst_archive: Path to generated .tar.gz. If omitted then use src_dir + .tar.gz
        regex: Optional regex of file paths to INCLUDE in the returned list
            This can either be a string, re.Pattern, or a callback function
            The tested path is the relative path to src_dir with forward slashes
            If a callback function is given, if the function returns True then the path is INCLUDED, else it is excluded
    Return:
        Path to generated .tar.gz
    """
    if not dst_archive:
        dst_archive = f'{os.path.dirname(os.path.abspath(src_dir))}/{os.path.basename(src_dir)}.tar.gz'

    if regex is not None:
        if isinstance(regex, str):
            regex = re.compile(regex)
            regex_func = regex.match
        elif isinstance(regex, re.Pattern):
            regex_func = regex.match
        else:
            regex_func = regex
    else:
        regex_func = lambda _: True # pylint: disable-unnecessary-lambda-assignment

    with tarfile.open(dst_archive, 'w:gz') as dst:
        for root, _, files in os.walk(src_dir):
            for fn in files:
                if fn == os.path.basename(dst_archive):
                    continue
                abs_path = os.path.join(root, fn)
                rel_path = os.path.relpath(abs_path, src_dir).replace('\\', '/')
                if not regex_func(rel_path):
                    continue
                dst.add(abs_path, arcname=rel_path)

    return dst_archive



def _extractall_patool(archive_path, output_dir, patool_path=None):
    archive_path = path.fullpath(archive_path)
    output_dir = path.fullpath(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Override the default tar command
    # so we can add the option: --force-local
    # This allows for running on Windows
    patoolib.programs.tar.extract_tar = _extract_tar

    try:
        patoolib.extract_archive(archive_path, interactive=False, outdir=output_dir)
    except patoolib.util.PatoolError as e:
        prepend_exception_msg(e, f'Failed to extract archive: {archive_path} to {output_dir}')
        if archive_path.endswith('.gz'):
            raise

        # This is extremely hacky but works sometimes...
        # If extraction failed, try changing the extension gz and run again
        old = archive_path
        base, _ = os.path.splitext(archive_path)
        archive_path = base + '.gz'
        try:
            os.remove(archive_path)
        except:
            pass
        shutil.copy2(old, archive_path)

        try:
            patoolib.extract_archive(archive_path, interactive=False, outdir=output_dir)
            return
        except patoolib.util.PatoolError:
            pass

        if 'could not find an executable program to extract format 7z' in f'{e}':
            msg = '\n\nIs 7zip installed on your computer? \n'
            if os.name == 'nt':
                msg += 'You can download and install it from here: https://www.7-zip.org/download.html'
            else:
                msg += 'You can install it with: sudo apt install p7zip-full'
            msg += '\n\n'
            append_exception_msg(e, msg)
        raise


def _extractall_zipfile(archive_path, output_dir):
    archive_path = path.fullpath(archive_path)
    output_dir = path.fullpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    ZipFile(archive_path).extractall(output_dir)


def _extractall_gzfile(archive_path, output_dir):
    archive_path = path.fullpath(archive_path)
    output_dir = path.fullpath(output_dir)

    with gzip.open(archive_path, 'rb') as f_in:
        fname, _ = _read_gzip_info(f_in)
        output_path = f'{output_dir}/{fname}'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def _extractall_tarfile(archive_path, output_dir):
    with tarfile.open(archive_path) as tar_file:
        def _is_within_directory(directory:str, target:str):
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)

            prefix = os.path.commonprefix([abs_directory, abs_target])

            return prefix == abs_directory

        def _safe_extract():
            # This fixes CVE-2007-4559: https://github.com/advisories/GHSA-gw9q-c7gh-j9vm,
            # which is a 15 year old bug in the Python tarfile package. By using extract() or extractall()
            # on a tarfile object without sanitizing input, a maliciously crafted .tar file could perform a directory path traversal attack.
            # We (Advanced Research Center at Trellix: https://www.trellix.com/) found at least one unsantized extractall() in your codebase and are providing a patch for you via pull request.
            # The patch essentially checks to see if all tarfile members will be extracted safely and throws an exception otherwise.
            # We encourage you to use this patch or your own solution to secure against CVE-2007-4559.
            # Further technical information about the vulnerability can be found in this blog:
            # https://www.trellix.com/en-us/about/newsroom/stories/research/tarfile-exploiting-the-world.html.
            for member in tar_file.getmembers():
                member_path = os.path.join(output_dir, member.name)
                if not _is_within_directory(output_dir, member_path):
                    raise RuntimeWarning(f"Attempted path traversal in TAR file: {archive_path}, archive path {member_path} not within {output_dir}")

            tar_file.extractall(output_dir)

        _safe_extract()



def _extractnested_archive(
    archive_path:str,
    output_dir:str,
    extract_nested:bool,
    remove_root_dir:bool
):
    ext = path.extension(archive_path)
    if not ext:
        raise Exception(f'Archive path: {archive_path} does not have a valid file extension')
    ext = '.' + ext

    tmp_dir = path.create_tempdir('tmp_archives/' + os.path.basename(archive_path).replace(ext, ''))
    extract_archive(archive_path, tmp_dir, clean_dest_dir=True)

    if extract_nested:
        nested_archive_path = None
        for root, _, files in os.walk(tmp_dir):
            if nested_archive_path is not None:
                break
            for fn in files:
                if fn.endswith(patoolib.ArchiveFormats + ('gz', 'bz', 'bz2')):
                    nested_archive_path = os.path.join(root, fn)
                    break

        if nested_archive_path is None:
            raise Exception(f'No nested archive found in {archive_path}')


        ext = path.extension(archive_path)
        nested_tmp_dir = tmp_dir + '/' + os.path.basename(nested_archive_path).replace(ext, '')
        extract_archive(nested_archive_path, nested_tmp_dir, clean_dest_dir=False)

        nested_src_dir = None
        for root, _, files in os.walk(nested_tmp_dir):
            if len(files) > 0:
                nested_src_dir = root
                break
    else:
        nested_src_dir = None
        for fn in os.listdir(tmp_dir):
            p = f'{tmp_dir}/{fn}'
            if os.path.isfile(p) or (os.path.isdir(p) and nested_src_dir is not None):
                raise Exception('Archive does not contain a single root directory')
            nested_src_dir = p


    if nested_src_dir is not None:
        path.copy_directory(nested_src_dir, output_dir)





# This overrides the default function in:
# patoolib.programs.tar
# It adds: "--force-local" to the command-line so that it can run on Windows
def _extract_tar (archive, compression, cmd, verbosity, interactive, outdir):
    """Extract a TAR archive."""
    cmdlist = [cmd, '--extract', '--force-local']
    patoolib.programs.tar.add_tar_opts(cmdlist, compression, verbosity)
    cmdlist.extend(["--file", archive, '--directory', outdir])
    return cmdlist


def _read_gzip_info(gzipfile: gzip.GzipFile) -> tuple:
    """Read the metadata from a gz file

    Returns:
    tuple(filename, size)
    """
    gf = gzipfile.fileobj
    pos = gf.tell()

    # Read archive size
    gf.seek(-4, 2)
    size = struct.unpack('<I', gf.read())[0]

    gf.seek(0)
    magic = gf.read(2)
    if magic != b'\037\213':
        raise IOError('Not a gzipped file')

    _, flag, _ = struct.unpack("<BBIxx", gf.read(8))

    if not flag & gzip.FNAME:
        # Not stored in the header, use the filename sans .gz
        gf.seek(pos)
        fname = gzipfile.name
        if fname.endswith('.gz'):
            fname = fname[:-3]
        return fname, size

    if flag & gzip.FEXTRA:
        # Read & discard the extra field, if present
        gf.read(struct.unpack("<H", gf.read(2)))

    # Read a null-terminated string containing the filename
    fname_bytes = bytearray()
    while True:
        s = gf.read(1)
        if not s or s==b'\000':
            break
        fname_bytes.extend(s)

    gf.seek(pos)
    fname = fname_bytes.decode('utf-8')
    return fname, size
