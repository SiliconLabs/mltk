import re


startup_file_re = re.compile(r'.*\/Source\/startup_.*\.c')



def should_patch_file(path: str) -> object:
    if startup_file_re.match(path):
        return dict(func=process_startup_file, state=0)

    return None

def process_file_line(lineno: int, line: str, arg: object) -> str:
    return arg['func'](lineno, line, arg)


def process_startup_file(lineno: int, line: str, arg: object) -> str:
    if 'Patched by MLTK' in line:
        return None

    if arg['state'] == 0 and line.strip() == 'void Default_Handler(void)':
        arg['state'] = 1
        line = '// Patched by MLTK\n'
        line += '#if defined __has_include\n'
        line += '#  if __has_include ("mltk_fault_handler.h")\n'
        line += '#    include "mltk_fault_handler.h"\n'
        line += '#  endif\n'
        line += '#endif\n'
        line += '#ifndef MLTK_ADD_FAULT_HANDLER\n'
        line += '#define MLTK_ADD_FAULT_HANDLER() while (true){}\n'
        line += '#endif\n\n'
        line += 'void Default_Handler(void)\n'

    elif arg['state'] == 1 and  line.strip() == '{':
        arg['state'] = 2
    elif arg['state'] == 2 and line.strip() == 'while (true) {':
        arg['state'] = 3
        line  = '  MLTK_ADD_FAULT_HANDLER();\n'
        line += '  // while (true)\n  {\n'

    return line


