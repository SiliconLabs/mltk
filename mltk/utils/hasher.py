"""Data hashing utilities

See the source code on Github: `mltk/utils/hasher.py <https://github.com/siliconlabs/mltk/blob/master/mltk/utils/hasher.py>`_
"""
from typing import Union
import os
import hashlib


def generate_hash(*args) -> str:
    """Generate an MD5 hash of the given Python objects"""
    md5 = hashlib.md5()
    hash_object(*args, hasher=md5)
    return md5.hexdigest().lower()


def hash_file(
    path: str,
    algorithm: Union[str,'hashlib._Hash'] = 'md5',
    include_filename: bool = False,
) -> str:
    """Generate a hash of the given file"""
    if not os.path.exists(path):
        return None

    if isinstance(algorithm, str):
        algorithm = algorithm.lower()
        if algorithm in ('sha256', 'sha2'):
            hasher = hashlib.sha256()
        elif algorithm in ('sha128', 'sha1'):
            hasher = hashlib.sha1()
        elif algorithm == 'md5':
            hasher = hashlib.md5()
        else:
            raise ValueError('Hash algorithm must be md5, sha1, or sha256')
    elif hasattr(algorithm, 'update') and hasattr(algorithm, 'hexdigest'):
        hasher = algorithm
    else:
        raise ValueError('"algorithm argument must be the name of a hash algorithm or a hashlib._Hash instance')

    with open(path, 'rb') as f:
        if include_filename:
            hasher.update(path.encode('utf-8'))

        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)

    return hasher.hexdigest().lower()



def hash_object(*objects, hasher=None):
    """Hash the given object(s) and return the hashlib hasher instance

    If no hasher argument is provided, then automatically created
    a hashlib.md5()
    """
    if hasher is None:
        hasher = hashlib.md5()

    for obj in objects:
        if isinstance(obj, dict):
            for key, value in obj.items():
                hash_object(key, hasher=hasher)
                hash_object(value, hasher=hasher)
            continue
        elif isinstance(obj, (list,tuple)):
            for e in obj:
                hash_object(e, hasher=hasher)
            continue
        elif isinstance(obj, (bytes,bytearray)):
            hasher.update(obj)
            continue
        elif isinstance(obj, str):
            hasher.update(obj.encode('utf-8'))
            continue
        else:
            hasher.update(f'{obj}'.encode('utf-8'))

    return hasher
