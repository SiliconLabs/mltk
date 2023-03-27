import re
import os
import argparse
import sys
from typing import List, Iterator, Tuple

CURDIR = os.path.abspath(os.path.dirname(__file__))
MLTK_CPP_DIR = os.path.normpath(f'{CURDIR}/../..').replace('\\', '/')
MLTK_PYTHON_DIR = os.path.normpath(f'{CURDIR}/../../../mltk').replace('\\', '/')



def main():
    parser = argparse.ArgumentParser(description='Find a CMakeList.txt containing the given target and print its directory')
    parser.add_argument('target', help='CMake target to search')
    parser.add_argument('--paths', help='Additional search paths')

    args = parser.parse_args()

    search_dirs = []
    if args.paths:
        search_dirs.extend(x.strip() for x in args.paths.split(';') if x.strip())
    search_dirs.append(MLTK_CPP_DIR)
    search_dirs.append(MLTK_PYTHON_DIR)

    target_re = re.compile(args.target)
    add_executable_re = re.compile(r'add_executable\s*\(\s*([\w_]+)')
    add_library_re = re.compile(r'add_library\s*\(\s*([\w_]+)')
    add_custom_target_re = re.compile(r'add_custom_target\s*\(\s*([\w_]+)')
    project_re = re.compile(r'project\s*\(\s*([\w_]+)')

    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            raise Exception(f'Search directory does not exist: {search_dir}')

        search_depth=10

        for root, _, files in walk_with_depth(search_dir, depth=search_depth):
            root = root.replace('\\', '/')

            # Do not search the mltk/cpp/tools directory
            if root.startswith(f'{MLTK_CPP_DIR}/tools'):
                continue

            for fn in files:
                if fn.lower() == 'cmakelists.txt':
                    cmake_path = f'{root}/{fn}'

                    with open(cmake_path, 'r') as f:
                        contents = f.read().replace('\r', '').replace('\n', '')

                    _find_target(add_executable_re, root, contents, target_re)
                    _find_target(add_library_re, root, contents, target_re)
                    _find_target(project_re, root, contents, target_re)
                    _find_target(add_custom_target_re, root, contents, target_re)

                elif fn.lower().endswith('.cmake') and fn.lower().startswith('find'):
                    target = fn[len('find'):-len('.cmake')]
                    if target_re.match(target):
                        _print_result(target, root)


    search_paths_str = "\n".join(search_dirs)
    sys.stdout.write(f'Failed to find target: {args.target}, search paths (max depth {search_depth}):\n{search_paths_str}\n')
    sys.exit(-1)



def _find_target(regex: re.Pattern, target_dir: str, contents: str, target: re.Pattern):
    matches = regex.findall(contents)
    if not matches:
        return
    for match in matches:
        if target.match(match):
            _print_result(match, target_dir)



def _print_result(target, targe_dir):
    sys.stdout.write(f'{target};{targe_dir}\n')
    sys.exit()


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
        if depth and num_sep + depth <= num_sep_this:
            del dirs[:]



if __name__ == '__main__':
    main()


