import re
import os
import argparse
import sys


CURDIR = os.path.abspath(os.path.dirname(__file__))
MLTK_CPP_DIR = os.path.abspath(f'{CURDIR}/..').replace('\\', '/')


def main():
    parser = argparse.ArgumentParser(description='Find a CMakeList.txt containing the given target and print its directory')
    parser.add_argument('target', help='CMake target to search')
    parser.add_argument('--paths', help='Additional search paths')

    args = parser.parse_args()

    search_dirs = []
    if args.paths:
        search_dirs.extend(args.paths.split(';'))
    search_dirs.append(MLTK_CPP_DIR)

    target_re = re.compile(args.target)
    add_executable_re = re.compile(r'add_executable\s*\(\s*([\w_]+)')
    add_library_re = re.compile(r'add_library\s*\(\s*([\w_]+)')
    add_custom_target_re = re.compile(r'add_custom_target\s*\(\s*([\w_]+)')
    project_re = re.compile(r'project\s*\(\s*([\w_]+)')

    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            raise Exception(f'Bunk dir: {search_dir}')

        for root, _, files in os.walk(search_dir):
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
    print(f'Failed to find target: {args.target}, search paths:\n{search_paths_str}')
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



if __name__ == '__main__':
    main()


