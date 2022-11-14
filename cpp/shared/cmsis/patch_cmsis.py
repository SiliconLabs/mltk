

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
        line += '#  define memcpy __builtin_memcpy\n\n'
        line += '#ifdef __cplusplus\n'
        line += 'namespace std {\n\n'
        line += 'static inline void* __builtin_memset(void* b, int c, long int l)\n'
        line += '{\n'
        line += '  return ::__builtin_memset(b, c, l);\n'
        line += '}\n\n'
        line += 'static inline void* __builtin_memcpy(void* a, const void* b, long int l)\n'
        line += '{\n'
        line += '  return ::__builtin_memcpy(a, b, l);\n'
        line += '}\n\n'
        line += '} // namespace std\n'
        line += '#endif // __cplusplus\n'
        line += '#endif // CMSIS_FORCE_BUILTIN_FUNCTIONS\n\n'
    return line


def _check_if_already_patched(line):
    if '// Patched by MLTK' in line:
        return None 
    else:
        return '// Patched by MLTK\n' + line 