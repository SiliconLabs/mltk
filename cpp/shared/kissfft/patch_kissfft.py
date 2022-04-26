
# This script does 1 thing:
#

# 1. tensorflow/lite/micro/compatibility.h:
#    There seems to be a build error with in TF_LITE_REMOVE_VIRTUAL_DELETE
#    Update the macro to ensure the "delete" operator is public



def should_patch_file(path: str) -> object:
    if path.endswith('kiss_fftr.c'):
        return dict(state=0)

    return None 


def process_file_line(lineno: int, line: str, arg: object) -> str:
    if arg['state'] == 0 and line.strip() == '// Patched by MLTK':
        return None

    if arg['state'] == 0 and 'struct kiss_fftr_state' in line:
        arg['state'] = 1
        l = '// Patched by MLTK\n'
        l += '#define fprintf(...)\n\n'
        l += line
        return l

    return line
