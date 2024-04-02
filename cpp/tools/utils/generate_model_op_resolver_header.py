
import os
import argparse
import json
from typing import Union

from mltk.core.model import load_tflite_model
from mltk.core.tflite_model import TfliteModel
from mltk.utils.path import fullpath
from mltk.utils.hasher import hash_file
from mltk.utils.python import prepend_exception_msg
from mltk import cli


def generate_model_op_resolver_header(
    model: Union[str, TfliteModel],
    output:str=None,
    obj_name='MyOpResolver',
    prepend_header=True
) -> str:
    """Generate a model op resolver header file from a MLTK model or .tflite

    Args:
        model: Name of MLTK model or path to .tflite
        output: Path to generated output header
        obj_name: The name of the OpResolver class
    """

    if isinstance(model, str):
        try:
            tflite_path = load_tflite_model(
                model,
                print_not_found_err=True,
                return_tflite_path=True
            )
            tflite_model = TfliteModel.load_flatbuffer_file(tflite_path)
            tflite_path = fullpath(tflite_path)
        except Exception as e:
            prepend_exception_msg(e, f'Failed to load tflite model: {model}')
            raise
    elif isinstance(model, TfliteModel):
        tflite_path = fullpath(model.path)
        tflite_model = model

    if output:
        output = fullpath(output)
        old_generation_details = None
        generation_args_path = f'{output}.json'
        generation_details = dict(
            tflite_path=tflite_path,
            tflite_hash=hash_file(tflite_path),
            output=output,
            obj_name=obj_name,
            prepend_header=prepend_header
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

    layer_opcodes = set()
    for layer in tflite_model.layers:
        layer_opcodes.add(layer.opcode_str)

    out = ''
    if prepend_header:
        out += '#pragma once\n'
        out += f'// This was generated from {model}\n\n'
        out += '#include "tensorflow/lite/micro/compatibility.h"\n'
        out += '#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"\n\n'
    
    out += f'class {obj_name} : public tflite::MicroMutableOpResolver<{len(layer_opcodes)}> {{\n'
    out += '  public:\n'
    out += f'    {obj_name}() {{\n'
    for opcode in layer_opcodes:
        func_name = opcode.replace('_', ' ').title().replace(' ',  '').replace('1d', '1D').replace('2d', '2D').replace('3d', '3D')
        out += f'      Add{func_name}();\n'
    out += '    }\n\n'
    out += '  private:\n'
    out += '    TF_LITE_REMOVE_VIRTUAL_DELETE\n'
    out += '};\n\n'

    if output:
        with open(output, 'w') as f:
            f.write(out)

        with open(generation_args_path, 'w') as f:
            json.dump(generation_details, f, indent=3)

    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a model op resolver header file from a MLTK model or .tflite')
    parser.add_argument('model', help='Name of MLTK model or path to .tflite')
    parser.add_argument('--output', default='generated_op_resolver.hpp', help='Path to generated output header')
    parser.add_argument('--name', default='MyOpResolver', help='Name of custom op resolver class')

    args = parser.parse_args()
    cli.get_logger(verbose=True)
    try:
        generate_model_op_resolver_header(
            model=args.model,
            output=args.output,
            obj_name=args.name,
        )
        
        cli.print_info(f'Generated {args.output}\nfrom {args.model}')

    except Exception as _ex:
        cli.handle_exception('Failed to generate model op resolver header', _ex)