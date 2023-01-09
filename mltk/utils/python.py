"""Common Python utilities

See the source code on Github: `mltk/utils/python.py <https://github.com/siliconlabs/mltk/blob/master/mltk/utils/python.py>`_
"""

import collections
import sys
import os
import re
import json
import logging
import importlib
import inspect
import time
import copy
from enum import Enum
from typing import Iterable, Any, Union

from mltk import MLTK_ROOT_DIR
from .logger import DummyLogger, make_filelike
from .shell_cmd import run_shell_cmd
from .path import fullpath


SHORT_VERSION = '.'.join([str(x) for x in sys.version_info[:2]])
"""The Python version as <major>.<minor>
e.g.: 3.9
"""

def _defaultdict_not_found():
    return None


def DefaultDict(d: dict = None, **kwargs) -> collections.defaultdict:
    """Creates a directory that returns None if a key does not exist
    NOTE: Nested dictionaries are also updated to a defaultdict
    """

    def _convert_to_default_dict(obj):
        if isinstance(obj, dict):
            obj = DefaultDict(obj)
        elif isinstance(obj, list):
            for i, x in enumerate(obj):
                obj[i] = _convert_to_default_dict(x)
        return obj

    if d is not None:
        kwargs.update(d)

    for key, value in kwargs.items():
        kwargs[key] = _convert_to_default_dict(value)


    return collections.defaultdict(_defaultdict_not_found, kwargs)


class DictObject(dict):
    """Standard Python dictionary that allows for accessing entries as object properties, e.g.:

    my_dict_obj = DictObject({'foo': 1, 'bar': False})

    # Both lines do the same thing
    foo = my_dict_obj.foo
    foo = my_dict_obj['foo']

    my_dict_obj.bar = True
    my_dict_obj['bar'] = True

    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(e)

    def __setattr__(self, name, value):
        self[name] = value



def merge_dict(destination: dict, source: dict, copy_destination=False) -> dict:
    """Merge the source dictionary into the destination and return the destination"""
    if copy_destination:
        destination = copy.deepcopy(destination)

    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = destination.setdefault(key, {})
            merge_dict(node, value)
        else:
            destination[key] = value

    return destination


def as_list(obj: Any, split: str=None) -> list:
    """Convert the given object to a list

    - If obj is None, then return empty list
    - If obj is a string, If the `split` argument is given then return obj.split(split) else just wrap the string in a list
    """
    if obj is None:
        return []
    elif isinstance(obj, list):
        return obj
    elif isinstance(obj, str):
        return [obj] if not split else [x.strip() for x in obj.split(split)]
    elif isinstance(obj, collections.abc.Iterable):
        return [x for x in obj]
    else:
        return [obj]


def flatten_list(l : Iterable) -> list:
    """Flatten the given iterable object to a list"""
    retval = []
    for x in l:
        try:
            iter(x)
        except TypeError:
            is_iterable = False
        else:
            is_iterable = True

        if is_iterable:
            retval.extend(flatten_list(x))
        else:
            retval.append(x)
    return retval


def list_rindex(lst: Iterable, value: Any) -> int:
    """Reverse find element index that is given value"""
    for i, v in enumerate(reversed(lst)):
        if v == value:
            return len(lst) - i - 1  # return the index in the original list
    return -1


def contains_class_type(l: Iterable, cls: Any) -> bool:
    """Return if the given list contains a class with the given type"""
    for e in l:
        if isinstance(e, cls):
            return True
    return False


def get_case_insensitive(value: str, l: Iterable) -> str:
    """Get the given string with case-insensitive comparsion"""
    if value is None:
        return None
    value = value.lower()
    for v in l:
        if v.lower() == value:
            return v
    return None


def is_true(arg) -> bool:
    """Return if the given argument is a True value"""
    if isinstance(arg, str):
        return arg.lower() in ('yes', 'true', 'on', '1')
    if isinstance(arg, bool):
        return arg
    if isinstance(arg, int):
        return arg != 0
    raise Exception(f'Invalid boolean arg: {arg}')

def is_false(arg) -> bool:
    """Return if the given argument is a False value"""
    if isinstance(arg, str):
        return arg.lower() in ('no', 'false', 'off', '0')
    if isinstance(arg, bool):
        return arg
    if isinstance(arg, int):
        return arg == 0
    raise Exception(f'Invalid boolean arg: {arg}')


def forward_method_kwargs(**kwargs) -> dict:
    """Return all the keyword-arguments of a method, excluding the 'self' argument"""
    retval = {}
    for key, value in kwargs.items():
        if key == 'self' or key.startswith('_'):
            continue
        elif key == 'kwargs':
            retval.update(value)
        else:
            retval[key] = value
    return retval


def prepend_exception_msg(e:Exception, msg:str) -> Exception:
    """Prepend a message to the given exception"""
    e.args = (msg, *e.args)
    all_str = True
    for x in e.args:
        try:
            str(x)
        except:
            all_str = False
            break

    # If every entry in the exception msg is a string
    # then make it look pretty by combining into a coma-separated string
    if all_str:
        s = ', '.join(str(x) for x in e.args)
        e.args = (s, )

    return e


def append_exception_msg(e:Exception, msg:str) -> Exception:
    """Append a message to the given exception"""
    e.args = (*e.args, msg)
    all_str = True
    for x in e.args:
        try:
            str(x)
        except:
            all_str = False
            break

    # If every entry in the exception msg is a string
    # then make it look pretty by combining into a coma-separated string
    if all_str:
        s = ', '.join(str(x) for x in e.args)
        e.args = (s, )
    return e


def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    gettrace = getattr(sys, 'gettrace', lambda : None)
    return gettrace() is not None


def notebook_is_active() ->  bool:
    """Return if script is executing in a IPython notebook (e.g. Jupyter notebook)"""
    for x in sys.modules:
        if x.lower() == 'ipykernel':
            return True
    return False


def install_pip_package(
    package:str,
    module_name:str=None,
    logger: logging.Logger=None,
    install_dir:str=None,
    upgrade=False,
    no_deps=False
):
    """Install the given pip package is necessary"""
    logger = logger or DummyLogger()

    if install_dir:
        install_dir = fullpath(install_dir)
        if install_dir not in sys.path:
            os.makedirs(install_dir, exist_ok=True)
            logger.info(f'Adding {install_dir} to sys.path')
            sys.path.append(install_dir)

    version_match = re.match(r'([\w\_\-]+)([=<>]).*', package)
    if not module_name:
        if version_match:
            module_name = package[:version_match.start(2)]
        else:
            module_name = package

    # Only try to import the module without running pip if no version is specified and upgrade=False
    if not upgrade and not version_match:
        try:
            importlib.import_module(module_name)
            return
        except:
            pass

    make_filelike(logger)
    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append('-U')
    if no_deps:
        cmd.append('--no-deps')
    cmd.append(package)
    if install_dir:
        cmd.extend(['--target', install_dir])

    logger.warning(f'Running cmd: {" ".join(cmd)}\n(This may take awhile, please be patient ...)')
    retcode, retval = run_shell_cmd(cmd, outfile=logger)
    if retcode != 0:
        raise Exception(f'Failed to install pip package: {package}, err:\n{retval}')


def import_module_at_path(path:str, reload=False):
    """Import the Python module at the given path and return the imported module
    """
    module_package = None
    path = fullpath(path)
    mltk_root_path = fullpath(MLTK_ROOT_DIR)

    # If the path is within the mltk directory
    # Then generate the module path relative to the mltk package
    if path.startswith(f'{mltk_root_path}/'):
        mltk_rel_path = os.path.relpath(path, mltk_root_path).replace('\\', '/')
        module_package = None
        module_name = mltk_rel_path.replace('.py', '').replace('/', '.')

    # Else if the path to an external .py script was provided
    elif path.endswith('.py'):
        path_dir = os.path.dirname(path).replace('\\', '/')
        module_name = os.path.basename(path).replace('.py', '')
        if os.path.exists(f'{path_dir}/__init__.py'):
            # Do a relative import if the module is in a parent package
            module_name = '.' + module_name
            parent_dir = os.path.dirname(path_dir)
            module_package = os.path.basename(path_dir)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
        else:
            # Otherwise, do an absolute import
            if path_dir not in sys.path:
                sys.path.insert(0, path_dir)

    # Else the path to external Python directory was provided
    else:
        if not os.path.exists(f'{path}/__init__.py'):
            raise Exception(f'Given path to directory: {path} does not contain a __init__.py file')

        parent_dir = os.path.dirname(path).replace('\\', '/')
        module_name = os.path.basename(path)
        if os.path.exists(f'{parent_dir}/__init__.py'):
            # Do a relative import if the module is in a parent package
            module_name = '.' + module_name
            module_package = os.path.basename(parent_dir)
            parent_dir = os.path.dirname(parent_dir).replace('\\', '/')
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
        else:
            # Otherwise, do an absolute import
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

    # If the module has already been imported
    # then reload it if necessary
    if module_name in sys.modules:
        if reload:
            return importlib.reload(sys.modules[module_name])
        else:
            return sys.modules[module_name]

    # Otherwise import the module
    return importlib.import_module(module_name, package=module_package)


def load_json_safe(path:str, *args, **kwargs) -> object:
    """Load a JSON file and ignoring any single-line, multi-line comments and trailing commas

    Args:
        path: Path to JSON file
        args, kwargs: Arguments to pass into json.loads
    Return:
        Loaded JSON object
    """
    RE_SINGLE_LINE_COMMENT = re.compile(r'("(?:(?=(\\?))\2.)*?")|(?:\/{2,}.*)')
    RE_MULTI_LINE_COMMENT = re.compile(r'("(?:(?=(\\?))\2.)*?")|(?:\/\*(?:(?!\*\/).)+\*\/)', flags=re.M|re.DOTALL)
    RE_TRAILING_COMMA = re.compile(r',(?=\s*?[\}\]])')

    with open(path, 'r') as fp:
        unfiltered_json_string = fp.read()

    filtered_json_string = RE_SINGLE_LINE_COMMENT.sub(r'\1', unfiltered_json_string)
    filtered_json_string = RE_MULTI_LINE_COMMENT.sub(r'\1', filtered_json_string)
    filtered_json_string = RE_TRAILING_COMMA.sub('', filtered_json_string)

    return json.loads(filtered_json_string, *args, **kwargs)


def find_object_key_with_value(
    obj:object,
    needle:object,
    throw_exception=False
) -> str:
    """Given an  class or class instance, search the
    attribute values of the object for the given "needle" and return its corresponding key.

    Note: If a class if given then it must be instantiable using a default constructor.

    Args:
        obj: Class or class instance
        needle: Class attribute value to find in class instance
        throw_exception: If true, throw an exception if the needle is not found, return 'none' otherwise
    Return:
        Lowercase key of found attribute value or "none" if value is not found
    """

    if inspect.isclass(obj) and not issubclass(obj, Enum):
        obj = obj()

    for key in dir(obj):
        if getattr(obj, key) == needle:
            return key.lower()

    if throw_exception:
        raise ValueError(f'{needle} not found in {obj}')

    return 'none'


def find_object_value_with_key(
    obj:object,
    needle:str,
    ignore_case=False,
    throw_exception=False
):
    """Given a class or class instance, search the
    attribute keys of the object for the given "needle" and return its corresponding value.

    NOTE: If a class if given then it must be instantiable using a default constructor (except of Enum classes).

    Args:
        obj: Class or class instance
        needle: Class attribute key to find in class instance
        ignore_case: Ignore the key's case if True
        throw_exception: If true, throw an exception if the needle is not found, return None otherwise
    Return:
        Value of found attribute key or None if key is not found
    """
    if needle is None:
        return None

    if inspect.isclass(obj) and not issubclass(obj, Enum):
        obj = obj()

    if ignore_case:
        needle = needle.lower()

    for key in dir(obj):
        if ignore_case:
            if key.lower() == needle:
                return getattr(obj, key)
        else:
            if key == needle:
                return getattr(obj, key)

    if throw_exception:
        raise ValueError(f'{needle} not found in {obj}')

    return None


def find_object_value_with_key_or_value(
    obj:object, needle:Union[str,object],
    ignore_case=False,
    throw_exception=False
):
    """Given a class or class instance, search the
    attribute keys and values of the object for the given "needle" and return its corresponding value.

    NOTE: If a class if given then it must be instantiable using a default constructor (except of Enum classes).

    Args:
        obj: Class or class instance
        needle: Class attribute key or value to find in class instance
        ignore_case: Ignore the key's case if True (needle must be a string)
        throw_exception: If true, throw an exception if the needle is not found, return None otherwise
    Return:
        Value of found attribute key or None if key/value is not found
    """
    if needle is None:
        return None

    if inspect.isclass(obj) and not issubclass(obj, Enum):
        obj = obj()

    needle_lower = None
    if ignore_case and isinstance(needle, str):
        needle_lower = needle.lower()

    for key in dir(obj):
        value = getattr(obj, key)
        if (needle_lower is None and key == needle) or \
            (needle_lower is not None and key.lower() == needle_lower) or \
            (value == needle):
            return value

    if throw_exception:
        raise ValueError(f'{needle} not found in {obj}')

    return None


def timeit(method):
    """Decorator to measure time it takes for method or function to execute"""
    def timed(*args, **kw):
        try:
            ts = time.time()
            return method(*args, **kw)
        finally:
            te = time.time()
            diff = (te - ts) * 1000
            print(f'{method.__name__} {diff:4f}ms')
    return timed