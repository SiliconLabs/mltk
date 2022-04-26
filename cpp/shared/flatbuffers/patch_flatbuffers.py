


def should_patch_file(path: str) -> object:
    return path.endswith('CMakeLists.txt') or path.endswith('base.h')

def process_file_line(lineno: int, line: str, arg: object) -> str:
    if line.strip() == 'project(FlatBuffers)':
        line = '# Patched by MLTK\n'
        line += 'cmake_policy(SET CMP0048 NEW)\n'
        line += 'project(FlatBuffers VERSION 1.0.0)\n'

    if line.strip() == '#ifndef FLATBUFFERS_LOCALE_INDEPENDENT':
        line = '// Patched by MLTK\n'
        line += '#define FLATBUFFERS_LOCALE_INDEPENDENT 0\n'
        line += '#ifndef FLATBUFFERS_LOCALE_INDEPENDENT\n'
    return line