
import os
import argparse
import json

from mltk.core.model import load_tflite_model
from mltk.core.tflite_model import TfliteModel
from mltk.utils.path import fullpath
from mltk.utils.hasher import hash_file
from mltk import cli


def generate_model_op_resolver_header(
    model: str,
    output: str,
    obj_name='MyOpResolver',
):
    """Generate a model op resolver header file from a MLTK model or .tflite

    Args:
        model: Name of MLTK model or path to .tflite
        output: Path to generated output header
        variable_name: Name of MicroMutableOpResolver object
        obj_name: The name of the OpResolver class
    """

    try:
        tflite_path = load_tflite_model(
            model,
            print_not_found_err=True,
            return_tflite_path=True
        )
    except Exception as e:
        cli.abort(msg=f'\n\nFailed to load tflite model, err: {e}\n\n')

    output = fullpath(output)
    old_generation_details = None
    generation_args_path = f'{output}.json'
    generation_details = dict(
        tflite_path=fullpath(tflite_path),
        tflite_hash=hash_file(tflite_path),
        output=output,
        obj_name=obj_name
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


    tflite_model = TfliteModel.load_flatbuffer_file(tflite_path)
    layer_opcodes = set()
    for layer in tflite_model.layers:
        layer_opcodes.add(layer.opcode_str)


    with open(output, 'w') as f:
        f.write('#pragma once\n')
        f.write(f'// This was generated from {model}\n\n')
        f.write( '#include "tensorflow/lite/micro/compatibility.h"\n')
        f.write( '#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"\n\n')
        f.write(f'class {obj_name} : public tflite::MicroMutableOpResolver<{len(layer_opcodes)}> {{\n')
        f.write( '  public:\n')
        f.write(f'    {obj_name}() {{\n')

        for opcode in layer_opcodes:
            func_name = opcode.replace('_', ' ').title().replace(' ',  '').replace('1d', '1D').replace('2d', '2D').replace('3d', '3D')
            f.write(f'      Add{func_name}();\n')
        f.write('    }\n\n')
        f.write('  private:\n')
        f.write('    TF_LITE_REMOVE_VIRTUAL_DELETE\n')
        f.write('};\n\n')

    with open(generation_args_path, 'w') as f:
        json.dump(generation_details, f, indent=3)

    cli.print_info(f'Generated {output}\nfrom {tflite_path}')


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
    except Exception as _ex:
        cli.handle_exception('Failed to generate model op resolver header', _ex)