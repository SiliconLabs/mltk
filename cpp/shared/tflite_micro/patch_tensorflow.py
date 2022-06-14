

def should_patch_file(path: str) -> object:
    if path.endswith('/micro/compatibility.h'):
        return dict(func=process_compatibility_h)

    if path.endswith('/micro/kernels/kernel_util.h'):
        return dict(func=process_kernel_util_h, state=0)

    if path.endswith('/micro/kernels/kernel_runner.h'):
        return dict(func=process_kernel_runner_h, state=0)

    if path.endswith('/micro/fake_micro_context.cc'):
        return dict(func=process_fake_micro_context_cc, state=0)

    if path.endswith('/micro/micro_allocator.cc'):
        return dict(func=process_micro_allocator_cc, state=0)

    if path.endswith('/micro/arena_allocator/simple_memory_allocator.cc'):
        return dict(func=process_simple_memory_allocator_cc, state=0)

    if path.endswith('/micro/memory_planner/greedy_memory_planner.cc'):
        return dict(func=process_greedy_memory_planner_cc, state=0)

    if path.endswith(('/micro/micro_interpreter.h', '/micro/micro_allocator.h')):
        return dict(func=process_header_visibility, state=0)

    if path.endswith('/kernels/kernel_util.cc'):
        return dict(func=process_kernel_util_cc, state=0)

    if path.endswith('/kernels/op_macros.h'):
        return dict(func=process_op_macros_h, state=0)
    
    if path.endswith('/c/builtin_op_data.h'):
        return dict(func=process_builtin_op_data_h, state=0)

    return None 


def process_file_line(lineno: int, line: str, arg: object) -> str:
    return arg['func'](lineno, line, arg)


def process_kernel_util_h(lineno: int, line: str, arg: object) -> str:
    if line.strip() == 'TFLITE_DCHECK(tensor != nullptr);':
        line  = '  // Patched by MLTK\n'
        line += '  // TFLITE_DCHECK(tensor != nullptr);\n'
        line += '  if(tensor == nullptr){ return nullptr; }\n'

    return line


def process_op_macros_h(lineno: int, line: str, arg: object) -> str:
    if arg['state'] == 0 and 'Patched by the MLTK' in line:
        return None

    if arg['state'] == 0 and line.strip() == '#endif  // TENSORFLOW_LITE_KERNELS_OP_MACROS_H_':
        arg['state'] = 1
        line = '// Patched by the MLTK\n'
        line += '#if !defined(__arm__) && defined(__cplusplus)\n'
        line += '#  include <stdexcept>\n'
        line += '#  undef TFLITE_ABORT\n'
        line += '#  define TFLITE_ABORT throw std::runtime_error("TF-Lite assertion failed");\n'
        line += '#endif // __cplusplus\n'
        line += '\n\n#endif  // TENSORFLOW_LITE_KERNELS_OP_MACROS_H_\n'

    return line

def process_compatibility_h(lineno: int, line: str, arg: object) -> str:
    if line.strip() == 'void operator delete(void* p) {}':
        line = 'public: void operator delete(void* p) {} // Patched by MLTK to ensure this operator is public\n'
    return line


def process_kernel_runner_h(lineno: int, line: str, arg: object) -> str:
    if arg['state'] == 0 and 'Patched by the MLTK' in line:
        return None

    if line.strip() == 'private:':
        line = ' public: // private:\n'

    if arg['state'] == 0 and 'static constexpr int kKernelRunnerBufferSize_ = 10000;' in line:
        arg['state'] = 1
        line  = '// Patched by the MLTK\n'
        line += '#ifdef __arm__\n'
        line += 'static constexpr int kKernelRunnerBufferSize_ = 32*1024;\n'
        line += '#else \n'
        line += 'static constexpr int kKernelRunnerBufferSize_ = 16*1024*1024;\n'
        line += '#endif\n'

    return line


def process_fake_micro_context_cc(lineno: int, line: str, arg: object) -> str:
    if arg['state'] == 0 and  'IsAllTempTfLiteTensorDeallocated()' in line:
        arg['state'] = 1
    elif arg['state'] == 1:
        arg['state'] = 2
        if 'Patched by MLTK' not in line:
            line = '  // Patched by MLTK\n'
            line += '  return true; // !allocated_tensor_count_;\n'

    return line


def process_micro_allocator_cc(lineno: int, line: str, arg: object) -> str:
    if 'if (PopulateTfLiteTensorFromFlatbuffer(model, tensor, tensor_index,' in line:
        if 'MLTK' not in line:
            return '  if(tensor == nullptr){ return nullptr; } if (PopulateTfLiteTensorFromFlatbuffer(model, tensor, tensor_index, // Patched by MLTK\n'

    if 'TfLiteTensor* tensor = AllocatePersistentTfLiteTensorInternal()' in line:
        if 'MLTK' not in line:
            return "  TfLiteTensor* tensor = AllocatePersistentTfLiteTensorInternal(); if(tensor == nullptr){ return nullptr; } // Patched by MLTK"

    if arg['state'] == 0 and  'void MicroAllocator::DeallocateTempTfLiteTensor(' in line:
        arg['state'] = 1
    elif arg['state'] == 1:
        arg['state'] = 2
        if 'Patched by MLTK' not in line:
            line =  '  // Patched by MLTK\n'
            line += '  if(tensor == nullptr){ return; } // TFLITE_DCHECK(tensor != nullptr);\n'

    return line 

def process_simple_memory_allocator_cc(lineno: int, line: str, arg: object) -> str:
    if arg['state'] == 0 and  'SimpleMemoryAllocator::IsAllTempDeallocated()' in line:
        arg['state'] = 1
    elif arg['state'] == 1:
        arg['state'] = 2
        if 'MLTK' not in line:
            return '// Patch by MLTK\n  return true;\n' + line

    return line




def process_greedy_memory_planner_cc(lineno: int, line: str, arg: object) -> str:
    if arg['state'] == 0 and line.strip() == '// Patched by MLTK':
        return None

    if arg['state'] == 0 and 'namespace tflite' in line:
        arg['state'] = 1
        line = '// Patched by MLTK\n'
        line += 'bool mltk_tflm_force_buffer_overlap = false;\n\n\n'
        line += 'namespace tflite {\n'
        return line

    if arg['state'] == 1 and 'bool GreedyMemoryPlanner::DoesEntryOverlapInTime(' in line:
        arg['state'] = 2

    if arg['state'] == 2 and '{' in line:
        arg['state'] = 3
        line += '\n'
        line += '  // Patched by MLTK\n'
        line += '  if(mltk_tflm_force_buffer_overlap) return false;\n\n'

    return line


def process_header_visibility(lineno: int, line: str, arg: object) -> str:
    if arg['state'] == 0 and 'Patched by the MLTK' in line:
        return None

    if arg['state'] == 0 and line.strip() ==  'private:':
        arg['state'] = 1
        line = '// Patched by the MLTK\n'
        line += 'public:\n'

    return line


def process_kernel_util_cc(lineno: int, line: str, arg: object) -> str:
    if arg['state'] == 0 and 'const double output_scale = static_cast<double>(output->params.scale);' in line:
        arg['state'] = 1

    elif arg['state'] == 1 and '// Patched by MLTK' in line:
        arg['state'] = 2
    elif arg['state'] == 1 and 'TF_LITE_ENSURE(context, scale_diff / output_scale <= 0.02)' in line:
        arg['state'] = 2
        line = '    // Patched by MLTK\n'
        line += '    // TF_LITE_ENSURE(context, scale_diff / output_scale <= 0.02);\n'

    return line


def process_builtin_op_data_h(lineno: int, line: str, arg: object) -> str:
    if arg['state'] == 0 and '#endif  // __cplusplus' in line:
        arg['state'] = 1

    elif arg['state'] == 1:
        arg['state'] = 2
        if '// Patched by MLTK' not in line:
            line = '// Patched by MLTK\n'
            line += '#ifndef __arm__\n'
            line += '#pragma pack(push,4)\n'
            line += '#define enum enum  __attribute__((packed))\n'
            line += '#endif\n\n'

    elif arg['state'] == 2 and '#endif  // __cplusplus' in line:
        arg['state'] = 3

    elif arg['state'] == 3:
        arg['state'] = 4
        if '// Patched by MLTK' not in line:
            line = '// Patched by MLTK\n'
            line += '#ifndef __arm__\n'
            line += '#pragma pack(pop)\n'
            line += '#undef enum\n'
            line += '#endif\n\n'

    return line

