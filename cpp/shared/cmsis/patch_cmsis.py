

def should_patch_file(path: str) -> object:
    if path.endswith('/arm_math_types.h'):
        return True
    
    return None 


def process_file_line(lineno: int, line: str, arg: object) -> str:
    if lineno == 0:
        return _check_if_already_patched(line)

    if line.strip() == '#define _ARM_MATH_TYPES_H_':
        line += '\n\n// Patched by MLTK\n'
        line += '#ifdef CMSIS_FORCE_BUILTIN_FUNCTIONS\n'
        line += '#  define memset __builtin_memset\n'
        line += '#  define memcpy __builtin_memcpy\n'
        line += '#endif\n\n'
    return line


def _check_if_already_patched(line):
    if '// Patched by MLTK' in line:
        return None 
    else:
        return '// Patched by MLTK\n' + line 