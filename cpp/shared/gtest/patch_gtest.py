


def should_patch_file(path: str) -> object:
    if path.endswith('custom/gtest.h'):
        return dict(func=process_custom_gtest_h, state=0)
    if path.endswith('custom/gtest-port.h'):
        return dict(func=process_custom_gtest_port_h, state=0)
    if path.endswith('internal/gtest-port.h'):
        return dict(func=process_internal_gtest_port_h, state=0)
    if path.endswith('src/gtest-filepath.cc'):
        return dict(func=process_gtest_filepath_cc, state=0)
    if path.endswith('gtest-internal-inl.h'):
        return dict(func=process_gtest_internal_inl_h, state=0)
    if path.endswith('gtest.cc'):
        return dict(func=process_gtest_cc, state=0)
    return None 

def process_file_line(lineno: int, line: str, arg: object) -> str:
    return arg['func'](lineno, line, arg)

def process_custom_gtest_h(lineno: int, line: str, context: dict) -> str:
    if context['state'] == 0 and 'GOOGLETEST_INCLUDE_GTEST_INTERNAL_CUSTOM_GTEST_H_' in line:
        context['state'] = 1

    elif context['state'] == 1:
        context['state'] = 2
        s = '#define GOOGLETEST_INCLUDE_GTEST_INTERNAL_CUSTOM_GTEST_H_\n'
        s += '#include "./gtest-port.h"\n'
        return s

    elif context['state'] == 2:
        return None 

    return line


def process_custom_gtest_port_h(lineno: int, line: str, context: dict) -> str:
    patch_data=\
"""
#define GOOGLETEST_INCLUDE_GTEST_INTERNAL_CUSTOM_GTEST_PORT_H_
#ifdef __arm__
#undef GTEST_HAS_CLONE
#define GTEST_HAS_CLONE             0
#undef GTEST_HAS_EXCEPTIONS
#define GTEST_HAS_EXCEPTIONS        0
#undef GTEST_HAS_POSIX_RE
#define GTEST_HAS_POSIX_RE          0
#undef GTEST_HAS_RTTI
#define GTEST_HAS_RTTI              0
#undef GTEST_HAS_PTHREAD
#define GTEST_HAS_PTHREAD           0
#undef GTEST_HAS_STD_WSTRING
#define GTEST_HAS_STD_WSTRING       0
#undef GTEST_HAS_SEH
#define GTEST_HAS_SEH               0
#undef GTEST_HAS_STREAM_REDIRECTION
#define GTEST_HAS_STREAM_REDIRECTION 0
#undef GTEST_LINKED_AS_SHARED_LIBRARY
#define GTEST_LINKED_AS_SHARED_LIBRARY 0
#undef GTEST_CREATE_SHARED_LIBRARY
#define GTEST_CREATE_SHARED_LIBRARY 0
#undef GTEST_HAS_TR1_TUPLE
#define GTEST_HAS_TR1_TUPLE         0
#undef GTEST_HAS_DEATH_TEST

#endif // __arm__
"""

    if context['state'] == 0 and 'GOOGLETEST_INCLUDE_GTEST_INTERNAL_CUSTOM_GTEST_PORT_H_' in line:
        context['state'] = 1

    elif context['state'] == 1:
        context['state'] = 2
        if '__arm__' in line:
            return None 
        return patch_data

    elif context['state'] == 2:
        return None 

    return line




def _patch_line(line, new_value):
    new_line = '#ifdef __arm__   // Patched by MLTK\n'
    new_line += new_value + ' // Patched by MLTK\n'
    new_line +='#else  // Patched by MLTK\n'
    new_line += line
    new_line +='#endif // Patched by MLTK\n'
    return new_line

def _check_if_already_patched(line):
    if '// Patched by MLTK' in line:
        return None 
    else:
        return '// Patched by MLTK\n' + line 



def process_internal_gtest_port_h(lineno: int, line: str, context: dict) -> str:
    if lineno == 0:
        return _check_if_already_patched(line)


    if line.strip().startswith('inline int FileNo(FILE* file)'):
        line = _patch_line(line, 'inline int FileNo(FILE* file){return -1;}')
    
    elif line.strip().startswith('inline char* StrDup(const char* src)'):
        line = _patch_line(line, 'inline char* StrDup(const char* src){return nullptr;}')
    
    elif line.strip().startswith('inline FILE* FDOpen(int fd, const char* mode)'):
        line = _patch_line(line, 'inline FILE* FDOpen(int fd, const char* mode){return nullptr;}')

    return line


def process_gtest_filepath_cc(lineno: int, line: str, context: dict) -> str:
    if lineno == 0:
        return _check_if_already_patched(line)
    
    if line.strip() == 'int result = mkdir(pathname_.c_str(), 0777);':
        line = _patch_line(line, 'int result = -1;')
    
    elif line.strip() == 'char* result = getcwd(cwd, sizeof(cwd));':
        line = _patch_line(line, 'char* result = nullptr;')
    
    return line


def process_gtest_internal_inl_h(lineno: int, line: str, context: dict) -> str:
    if lineno == 0:
        return _check_if_already_patched(line)
        

    if context['state'] == 0 and line.strip() == 'if (original_working_dir_.IsEmpty()) {':
        context['state'] = 1
        line = '#ifndef __arm__ // Patched by MLTK\n' + line
    
    elif context['state'] == 1:
        if line.strip() == '}':
            context['state'] = 2 
            line = line + '#endif\n'

    elif context['state'] == 2:
        return None
    
    return line

def process_gtest_cc(lineno: int, line: str, context: dict) -> str:
    if lineno == 0:
        context['state'] = {'force_color': 0, 'color': 0, 'flags': 0, 'config': 0, 'output': 0, 'unstatic printf': 0, 'update_color': 0}
        return _check_if_already_patched(line)

    state = context['state']

    if state['force_color'] == 0 and line.strip() == 'static const bool in_color_mode =':
        state['force_color'] = 1
        return 'static const bool in_color_mode = true; // Patched by MLTK\n' 

    if state['unstatic printf'] == 0 and line.strip().startswith('static void ColoredPrintf('):
        state['unstatic printf'] = 1
        return 'void ColoredPrintf(GTestColor color, const char* fmt, ...) { // Patched by MLTK\n' 

    if state['color'] == 0 and line.strip() == 'if (!use_color) {':
        state['color'] = 1
        return '(void)use_color;\n#ifndef __arm__ // Patched by MLTK\n' + line
    elif state['color'] == 1 and line.strip() == '}':
        state['color'] = 2
        return line + '#endif\n'


    if state['flags'] == 0 and line.strip() == 'ParseGoogleTestFlagsOnly(argc, argv);':
        state['flags'] = 1
        return '#ifndef __arm__ // Patched by MLTK\n' + line 
    elif state['flags'] == 1 and line.strip() == 'GetUnitTestImpl()->PostFlagParsingInit();':
        state['flags'] = 2
        return line + '#endif\n'
    

    if state['config'] == 0 and line.strip() == 'void UnitTestImpl::ConfigureXmlOutput() {':
        state['config'] = 1
        return line + '#ifndef __arm__ // Patched by MLTK\n'
    elif state['config'] == 1 and line.startswith('}'):
        state['config'] = 2
        return '#endif\n' + line

    if state['config'] != 1:
        if state['output'] == 0 and line.strip() == 'const std::string& output_format = UnitTestOptions::GetOutputFormat();':
            state['output'] = 1
            return '#ifndef __arm__ // Patched by MLTK\n' + line
        elif state['output'] == 1 and line.startswith('  }'):
            state['output'] = 2
            return line + '#endif\n'

    if state['update_color'] == 0 and line.strip() == 'namespace internal {':
        state['update_color'] = 1
    elif state['update_color'] == 1:
        if line.strip() == 'namespace {':
            # Remove the empty namespace around:
            # enum class GTestColor { kDefault, kRed, kGreen, kYellow }
            state['update_color'] = 2 
            return '// namespace { // Patched by MLTK\n'
        else:
            state['update_color'] = 0
    elif state['update_color'] == 2:
        state['update_color'] = 3
    elif state['update_color'] == 3:
        state['update_color'] = 4
        return '// } // Patched by MLTK\n' 


    return line