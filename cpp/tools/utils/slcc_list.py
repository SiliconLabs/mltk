import sys
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='Utility to list the includes and sources of a directory for populating a .slcc file')
    parser.add_argument('type', choices=['sources', 'headers'])
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('dir_path')

    args = parser.parse_args()

    def should_include_source(fn):
        if not fn.endswith(('.c', '.cc', '.cpp')):
            return False
        if fn.endswith('_test.cc'):
            return False

        return True

    def should_include_header(fn):
        if not fn.endswith(('.h', '.hpp')):
            return False

        return True


    lines = []
    if args.type == 'sources':
        if not args.recursive:
            for fn in os.listdir(args.dir_path):
                if should_include_source(fn):
                    lines.append(fn)
        else:
            for root, _, files in os.walk(args.dir_path):
                for fn in files:
                    if should_include_source(fn):
                        p = f'{root}/{fn}'
                        p = os.path.relpath(p, args.dir_path).replace('\\', '/')
                        lines.append(p)

        for p in sorted(lines):
            print(f'  - path: {p}')

    elif args.type == 'headers':
        if not args.recursive:
            for fn in os.listdir(args.dir_path):
                if should_include_header(fn):
                    lines.append(fn)
        else:
            for root, _, files in os.walk(args.dir_path):
                for fn in files:
                    if should_include_header(fn):
                        p = f'{root}/{fn}'
                        p = os.path.relpath(p, args.dir_path).replace('\\', '/')
                        lines.append(p)

        for p in sorted(lines):
            print(f'      - path: {p}')



if __name__ == '__main__':
    main()