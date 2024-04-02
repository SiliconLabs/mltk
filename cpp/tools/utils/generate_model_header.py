
import os
import argparse
import json
import hashlib
import logging

from mltk.core.model import load_tflite_model
from mltk.core.tflite_model_parameters import TfliteModelParameters
from mltk.core.tflite_micro import TfliteMicro
from mltk.utils.bin2header import bin2header
from mltk.utils.path import fullpath, create_tempdir
from mltk.utils.hasher import hash_file
from mltk.utils.python import as_list
from mltk import cli
from cpp.tools.utils.generate_model_op_resolver_header import generate_model_op_resolver_header

def generate_model_header(
    model: str,
    output: str,
    variable_name='mltk_model_flatbuffer',
    variable_attributes:str=None,
    length_variable_name='mltk_model_flatbuffer_length',
    accelerator:str=None,
    model_memory_section:str=None,
    runtime_memory_section:str=None,
    runtime_memory_size:int=-1,
    generate_op_resolver:bool=False, 
    depends:str=None,
    settings_file:str=None,
    platform:str=None,
):
    """Generate a model header file from a MLTK model or .tflite

    Args:
        model: Name of MLTK model or path to .tflite
        output: Path to generated output header
        variable_name: Name of C array
        variable_attributes: Attributes to prepend to C array variable
        length_variable_name: Name of C variable to hold length of C array
        accelerator: Name of accelerator for which to generate header
        memory_section: The memory section to place the model
        depends: semi-colon separated list of paths of additional dependency files (used to determine if the model is up-to-date)
        settings_file: Path to model settings file
    """

    try:
        tflite_model = load_tflite_model(
            model,
            print_not_found_err=True,
        )
        tflite_path = fullpath(tflite_model.path)
    except Exception as e:
        cli.abort(msg=f'\n\nFailed to load tflite model, err: {e}\n\n')

    
    output = fullpath(output)

    depends_paths = list(fullpath(x) for x in as_list(depends, split=';'))
    depends_hasher=hashlib.md5()
    for p in depends_paths:
        hash_file(p, algorithm=depends_hasher)
    if settings_file and os.path.exists(settings_file):
        hash_file(settings_file, algorithm=depends_hasher)

    depends_hash = depends_hasher.hexdigest().lower()

    old_generation_details = None
    generation_args_path = f'{output}.json'
    generation_details = dict(
        tflite_path=tflite_path,
        tflite_hash=hash_file(tflite_path),
        output=output,
        accelerator=accelerator,
        platform=platform,
        variable_name=variable_name,
        variable_attributes=variable_attributes,
        length_variable_name=length_variable_name,
        model_memory_section=model_memory_section,
        runtime_memory_section=runtime_memory_section,
        runtime_memory_size=runtime_memory_size,
        generate_op_resolver=generate_op_resolver,
        depends_hash=depends_hash
    )

    if os.path.exists(generation_args_path):
        try:
            with open(generation_args_path, 'r') as f:
                old_generation_details = json.load(f)
        except:
            pass

    if old_generation_details == generation_details:
        print(f'{os.path.basename(output)} up-to-date')
        return

    if accelerator and accelerator.lower() != 'cmsis':
        tflm_accelerator = TfliteMicro.get_accelerator(accelerator)
        if tflm_accelerator.supports_model_compilation:
            compilation_report_path = output + '-compilation_report.txt'
            tflite_model = tflm_accelerator.compile_model(
                tflite_model,
                report_path=compilation_report_path,
                logger=cli.get_logger(),
                settings_file=settings_file,
                platform=platform
            )
            tflite_path = f'{create_tempdir("tmp_models")}/{tflite_model.name}.{accelerator}.tflite'
            tflite_model.save(tflite_path, update_path=True)

    header_includes = ['<cstdint>']

    variable_attributes = variable_attributes or ''
    if model_memory_section:
        variable_attributes += f' __attribute__((section("{model_memory_section}")))'
    model_flatbuffer_header = bin2header(
        input=tflite_path,
        var_name=variable_name,
        length_var_name=length_variable_name,
        attributes=variable_attributes,
        prepend_header=False
    )

    if generate_op_resolver:
        name =  tflite_model.name
        op_resolver_name = ''.join((x.capitalize() for x in name.replace('.', '_').replace('-', '_').split('_'))) + 'OpResolver'
        model_op_resolver_header = generate_model_op_resolver_header(
            model=tflite_model.path,
            obj_name=op_resolver_name,
            prepend_header=False
        )
        header_includes.append('tensorflow/lite/micro/compatibility.h')
        header_includes.append('tensorflow/lite/micro/micro_mutable_op_resolver.h')
        model_op_resolver_header += f'\n {op_resolver_name} mltk_model_op_resolver;\n'
    else:
        header_includes.append('all_ops_resolver.h')
        model_op_resolver_header = "tflite::AllOpsResolver mltk_model_op_resolver;\n"


    memory_buffers_header = ''
    try:
        memory_spec = TfliteModelParameters.load_from_tflite_model(
            tflite_model, 
            tag=f'{accelerator}_memory_spec'
        )
    except:
        memory_spec = {}


    if memory_spec:
        model_buffer_sizes = memory_spec.get('sizes', [])
        model_buffer_sections = memory_spec.get('sections', [])
        model_buffer_names = memory_spec.get('names', [])
        model_buffer_count = len(model_buffer_sizes)
        size_var_names = []
        buffer_var_names = []
        
        for i in range(model_buffer_count):
            model_buffer_i_size = model_buffer_sizes[i]
            model_buffer_i_section = '.bss' if i >= len(model_buffer_sections) else model_buffer_sections[i]
            model_buffer_i_name = f'{i}' if i >= len(model_buffer_names) else model_buffer_names[i]
            size_var_names.append(f'mltk_model_buffer_{model_buffer_i_name}_size')
            buffer_var_names.append(f'mltk_model_buffer_{model_buffer_i_name}')
            memory_buffers_header += f'const int32_t {size_var_names[-1]} = {model_buffer_i_size};\n'
            memory_buffers_header += f'uint8_t __attribute__((section("{model_buffer_i_section}"))) {buffer_var_names[-1]}[{model_buffer_i_size}];\n'

        memory_buffers_header += f'const int32_t mltk_model_buffer_count = {model_buffer_count};\n'
        memory_buffers_header += f'const int32_t mltk_model_buffer_sizes[{model_buffer_count}] = {{ ' + ', '.join(size_var_names) + ' };\n'
        memory_buffers_header += f'uint8_t* mltk_model_buffers[{model_buffer_count}] = {{ ' + ', '.join(buffer_var_names) + ' };\n'

    elif runtime_memory_size is not None:
        runtime_memory_size = int(runtime_memory_size)
        runtime_memory_section = runtime_memory_section or '.bss'

        if runtime_memory_size > 0:
            memory_buffers_header += f'const int32_t mltk_model_runtime_memory_size = {runtime_memory_size};\n'
            memory_buffers_header += f'uint8_t __attribute__((section("{runtime_memory_section}"))) mltk_model_runtime_memory_buffer[{runtime_memory_size}];\n'
        else:
            memory_buffers_header += f'const int32_t mltk_model_runtime_memory_size = {runtime_memory_size};\n'
            memory_buffers_header += 'uint8_t* mltk_model_runtime_memory_buffer = nullptr;\n'

        memory_buffers_header += 'uint8_t* mltk_model_buffers[1] = { mltk_model_runtime_memory_buffer };\n'
        memory_buffers_header += 'const int32_t mltk_model_buffer_sizes[1] = { mltk_model_runtime_memory_size };\n'
        memory_buffers_header += 'const int32_t mltk_model_buffer_count = 1;\n'


    with open(output, 'w') as f:
        f.write('#pragma once\n\n')
        for header in header_includes:
            if not header.startswith("<"):
                header = f'"{header}"'
            f.write(f'#include {header}\n')

        f.write(f'\n{model_op_resolver_header}\n\n')
        f.write(f'{memory_buffers_header}\n\n')
        f.write(f'{model_flatbuffer_header}\n')

    with open(generation_args_path, 'w') as f:
        json.dump(generation_details, f, indent=3)

    cli.print_info(f'Generated {output}\nfrom {tflite_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a model header file from a MLTK model or .tflite')
    parser.add_argument('model', help='Name of MLTK model or path to .tflite')
    parser.add_argument('--output', default='generated_model.tflite.hpp', help='Path to generated output header')
    parser.add_argument('--name', default='mltk_model_flatbuffer', help='Name of model flatbuffer C array')
    parser.add_argument('--length-name', default='mltk_model_flatbuffer_length', help='Name of C variable to hold length of model flatbuffer data in bytes')
    parser.add_argument('--attributes', default=None, help='Attributes to prepend to model flatbuffer C array variable')
    parser.add_argument('--accelerator', default=None, help='Specific accelerator for which to generate model header')
    parser.add_argument('--platform', default=None, help='Specific platform to compile model')
    parser.add_argument('--model-memory-section', default=None, help='The memory section to place the model')
    parser.add_argument('--runtime-memory-section', default=None, help='The memory section to place the runtime buffer (aka tensor arena)')
    parser.add_argument('--runtime-memory-size', default='-1', help='The size in bytes of the runtime buffer (aka tensor arena)')
    parser.add_argument('--generate-op-resolver', default=False, action='store_true',  help='Generate the op resolve from the ops used in the given model')
    parser.add_argument('--depends', default=None, help='Semi-colon separated list of paths of additional dependency files (used to determine if the model is up-to-date)')
    parser.add_argument('--settings-file', default=None, help='Path to model settings file')

    args = parser.parse_args()
    
    logger = cli.get_logger(verbose=False)
    fh = logging.FileHandler(args.output.replace('.c', '') + '-log.txt', mode='w')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    try:
        generate_model_header(
            model=args.model,
            output=args.output,
            variable_name=args.name,
            length_variable_name=args.length_name,
            variable_attributes=args.attributes,
            accelerator=args.accelerator,
            platform=args.platform,
            model_memory_section=args.model_memory_section,
            runtime_memory_section=args.runtime_memory_section,
            runtime_memory_size=args.runtime_memory_size,
            generate_op_resolver=args.generate_op_resolver,
            depends=args.depends,
            settings_file=args.settings_file
        )
    except Exception as _ex:
        cli.handle_exception('Failed to generate model header', _ex)