import sys
import os

try:
    from grpc_tools import protoc
except ModuleNotFoundError:
    print('You must first install the Python package: "grpcio-tools"')
    print('pip install grpcio-tools')
    sys.exit(-1)


curdir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')


def generate():
    proto_path = f'{curdir}/proto'

    os.chdir(proto_path)
    print(f'Processing {proto_path}')
    protoc.main((
        '',
        '-I.',
        "--python_out=.",
        '--pyi_out=.',
        "--grpc_python_out=.",
        'download_run.proto'
    ))

if __name__ == '__main__':
    generate()