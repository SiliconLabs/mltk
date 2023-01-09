"""File path utilities

See the source code on Github: `mltk/utils/path.py <https://github.com/siliconlabs/mltk/blob/master/mltk/utils/path.py>`_
"""
import os
import time
import re
import glob
import shutil
import tempfile
import datetime
import subprocess
from pathlib import Path
from typing import Callable, List, Union, Iterator, Tuple

from .system import get_username


# This is the base directory used for temporary files
# It includes the the current user's name in the path
# to separate temp files for different user's that use the same tempdir
TEMP_BASE_DIR = os.environ.get('MLTK_TEMP_DIR', f'{tempfile.gettempdir()}/{get_username()}/mltk')


def fullpath(path : str, cwd:str=None) -> str:
    """Return the full, normalized path of the given path"""
    orig_cwd = os.getcwd()
    if cwd:
        os.chdir(cwd)

    try:
        path = os.path.expandvars(path)
        path = os.path.expanduser(path)
        path = os.path.normpath(path)
        path = os.path.abspath(path)
    finally:
        os.chdir(orig_cwd)

    # Return the filepath as it actually exists on the FS
    # If the path doesn't exist, then just return the normalized path
    path = get_actual_path(path) or path

    return path.replace('\\', '/')


def get_actual_path(path: str):
    """Return the file path as it actually appears on the FS (including upper/lower case)
    Return None if the path doesn't exist
    """
    try:
        dirs = path.split('\\')
        # disk letter
        test_name = [dirs[0].upper()]
        for d in dirs[1:]:
            test_name += ["%s[%s]" % (d[:-1], d[-1])]
        res = glob.glob('\\'.join(test_name))
        if not res:
            #File not found
            return None

        return res[0]
    except:
        return None


def extension(path : str) -> str:
    """Return the extension of the given path"""
    idx = path.rfind('.')
    if idx == -1:
        return ''

    if idx == 0:
        return ''

    return path[idx+1:]


def has_filename(path: str) -> bool:
    """Return if the given path has a filename with an extension or is only a directory path"""
    if not path:
        return False
    path = fullpath(path)
    last_slash_index = path.rfind('/')
    ext_index = path[last_slash_index+1:].rfind('.')
    return ext_index > 0


def create_dir(path:str):
    """Create the given path's directories"""
    path = fullpath(path)
    if has_filename(path):
        path = os.path.dirname(path)

    if not path:
        return

    os.makedirs(path, exist_ok=True)


def create_tempdir(subdir='') -> str:
    """Create a temporary directory as <temp dir>/<username>/mltk"""
    d = TEMP_BASE_DIR
    if subdir:
        subdir = subdir.replace('\\', '/')
        if not subdir.startswith('/'):
            d += f'/{subdir}'

    d = fullpath(d)
    os.makedirs(d, exist_ok=True)

    return d


def create_user_dir(suffix:str='', base_dir:str=None) -> str:
    """Create a user directory

    This creates the directory in one of the following base directories
    based on availability:
    - base_dir argument
    - OS environment variable: MLTK_CACHE_DIR
    - ~/.mltk
    - <user temp dir>/<username>/mltk

    Args:
        suffix: Optional suffix to append to the base directory
        base_dir: Optional base directory, default to MLTK_CACHE_DIR, ~/.mltk, or <user temp dir>/<user name>/mltk if omitted

    Returns:
        path to created directory

    """
    if suffix:
        suffix = suffix.replace('\\', '/')
        if not suffix.startswith('/'):
            suffix = f'/{suffix}'

    is_read_only = os.environ.get('MLTK_READONLY')
    user_dir = base_dir if base_dir else os.environ.get('MLTK_CACHE_DIR', '~/.mltk')
    user_dir = fullpath(user_dir + suffix)

    # If the MLTK_READONLY environment variable is set
    # then don't check if the user_dir directory is writable
    if not is_read_only:
        try:
            # Try to create the directory in MLTK_CACHE_DIR or ~/.mltk
            # if we have permission
            os.makedirs(user_dir, exist_ok=True)
            if not os.access(user_dir, os.W_OK):
                raise Exception()
        except:
            # Otherwise just create in the temp directory
            user_dir = create_tempdir(suffix)

    return user_dir


def get_user_setting(name:str, default=None):
    """Return the value of a user setting if it exists

    User settings are defined in the file:

    - Environment variable: MLTK_USER_SETTINGS_PATH
    - OR <user home>/.mltk/user_settings.yaml

    User settings include:

    - model_paths: list of directories to search for MLTK models
    - commander: Simplicity Commander options
      device: Device code
      serial_number: Adapter serial number
      ip_address: Adapter IP address

    See `settings_file <https://siliconlabs.github.io/mltk/docs/other/settings_file.html>`_

    """
    user_settings_path = fullpath(os.environ.get('MLTK_USER_SETTINGS_PATH', '~/.mltk/user_settings.yaml'))
    if not os.path.exists(user_settings_path):
        return default

    try:
        # Import the YAML package here
        # in-case it's not installed yet
        import yaml
        with open(user_settings_path, 'r') as fp:
            user_settings = yaml.load(fp, Loader=yaml.SafeLoader)
    except Exception as e:
        if not '_printed_user_settings_error' in globals():
            globals()['_printed_user_settings_error'] = True
            print(f'WARN: Failed to parse {user_settings_path}, err: {e}')
        return default

    if user_settings and name in user_settings:
        return user_settings[name]

    return default


def add_user_setting(name:str, value:object):
    """Add an entry to the user settings

    User settings are defined in the file:
    <user home>/.mltk/user_settings.yaml
    """
    # Import the YAML package here
    # in-case it's not installed yet
    import yaml

    user_settings_paths = fullpath('~/.mltk/user_settings.yaml')
    if os.path.exists(user_settings_paths):
        with open(user_settings_paths, 'r') as fp:
            user_settings = yaml.load(fp, Loader=yaml.SafeLoader)
    else:
        user_settings = dict()

    user_settings[name] = value

    with open(user_settings_paths, 'w') as f:
        yaml.dump(user_settings, f, Dumper=yaml.SafeDumper)



def remove_directory(path:str):
    """Remove the directory at the given path

    This will remove non-empty directories and retry a few times if necessary

    """
    if not path:
        return

    def _remove_dir(d):
        if os.name == 'nt':
            subprocess.check_output(['cmd', '/C', 'rmdir', '/S', '/Q', os.path.abspath(d)])
        else:
            subprocess.check_output(['rm', '-rf', os.path.abspath(d)])


    retries = 5
    while retries > 0 and os.path.exists(path):
        try:
            _remove_dir(path)
        except:
            pass
        finally:
            time.sleep(.001)
            retries -= 1

def clean_directory(path:str):
    """Remove all files within directory and subdirectories
    """
    for root, _, files in os.walk(path):
        for fn in files:
            p =  f'{root}/{fn}'
            for _ in range(3):
                try:
                    os.remove(p)
                    break
                except:
                    time.sleep(.001)


def copy_directory(src, dst, exclude_dirs=None):
    """Recursively copy a directory. Only copy files that are new or out-dated"""

    if exclude_dirs:
        for x in exclude_dirs:
            if src.replace('\\', '/').startswith(x.replace('\\', '/')):
                return

    if not os.path.exists(dst):
        os.makedirs(dst)

    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copy_directory(s, d, exclude_dirs=exclude_dirs)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)



def set_file_last_modified(file_path: str, dt: datetime.datetime = None):
    if dt is None:
        dt = datetime.datetime.fromisoformat('2000-01-01')
    dt_epoch = int(dt.timestamp())
    os.utime(file_path, (dt_epoch, dt_epoch))


def file_is_newer(source_path: str, other_path: str):
    if not os.path.exists(source_path) or not os.path.exists(other_path):
        return False
    return os.stat(source_path).st_mtime > os.stat(other_path).st_mtime


def file_is_in_use(file_path:str) -> bool:
    """Return if the file is currently opened"""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError

    try:
        path.rename(path)
    except PermissionError:
        return True
    else:
        return False


def recursive_listdir(
    base_dir:str,
    followlinks=True,
    regex:Union[str,re.Pattern,Callable[[str],bool]]=None,
    return_relative_paths:bool=False
) -> List[str]:
    """Return list of all files recursively found in base_dir

    Args:
        base_dir: The base directory to recursively search
        followlinks: IF true then follow symbolic links
        regex: Optional regex of file paths to INCLUDE in the returned list
            This can either be a string, re.Pattern, or a callback function
            If return_relative_paths=False then the tested path is the absolute path with forward slashes
            If return_relative_paths=True then the tested path is the path relative to the base_dir with forward slashes
            If a callback function is given, if the function returns True then the path is INCLUDED, else it is excluded
        return_relative_paths: If true then return paths relative to the base_dir, else return absolute paths

    Returns:
        List of file paths with forward slashes for directory delimiters
    """

    base_dir = fullpath(base_dir)

    if regex is not None:
        if isinstance(regex, str):
            regex = re.compile(regex)
            regex_func = regex.match
        elif isinstance(regex, re.Pattern):
            regex_func = regex.match
        else:
            regex_func = regex
    else:
        regex_func = lambda p: True

    retval = []
    for root, _, files in os.walk(base_dir, followlinks=followlinks):
        for fn in files:
            p = os.path.join(root, fn).replace('\\', '/')
            if return_relative_paths:
                p = os.path.relpath(p, base_dir).replace('\\', '/')
            if not regex_func(p):
                continue
            retval.append(p)

    return retval


def walk_with_depth(
    base_dir:str,
    depth=1,
    followlinks=True,
) -> Iterator[Tuple[str, List[str], List[str]]]:
    """Walk a directory with a max depth.

    This is similar to os.walk except it has an optional maximum directory depth that it will walk
    """
    base_dir = base_dir.rstrip(os.path.sep)
    assert os.path.isdir(base_dir)
    num_sep = base_dir.count(os.path.sep)
    for root, dirs, files in os.walk(base_dir, followlinks=followlinks):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + depth <= num_sep_this:
            del dirs[:]
