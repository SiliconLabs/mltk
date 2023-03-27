

def should_patch_file(path: str) -> object:
    if path.endswith('/unity_internals.h'):
        return dict(func=process_unity_internals_h, state=0)

    return None

def process_file_line(lineno: int, line: str, arg: object) -> str:
    return arg['func'](lineno, line, arg)


def process_unity_internals_h(lineno: int, line: str, arg: object) -> str:
    if 'Patched by MLTK' in line:
        return None

    if '#define isinf(n)' in line:
        line = '// Patched by MLTK\n// ' + line

    if '#define isnan(n)' in line:
        line = '// Patched by MLTK\n// ' + line

    return line