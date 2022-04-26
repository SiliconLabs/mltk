import sys

try:
    from ninja import BIN_DIR

    ninja_path = f'{BIN_DIR}/ninja'.replace('\\', '/')
    sys.stdout.write(f'{ninja_path};')
    sys.exit(0)
except Exception as e:
    sys.stdout.write(f'Failed to get ninja path, err: {e};')
    sys.exit(-1)