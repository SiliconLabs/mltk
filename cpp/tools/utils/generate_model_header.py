
import os
import argparse
import json

from mltk.core.model import load_mltk_model
from mltk.utils.bin2header import bin2header
from mltk.utils.path import fullpath
from mltk.utils.hasher import hash_file
from mltk import cli


def generate_model_header(
    model: str,
    output: str,
    variable_name='MODEL_DATA',
    variable_attributes:str=None,
    length_variable_name='MODEL_DATA_LENGTH',
):
    """Generate a model header file from a MLTK model or .tflite
    
    Args:
        model: Name of MLTK model or path to .tflite
        output: Path to generated output header
        variable_name: Name of C array
        variable_attributes: Attributes to prepend to C array variable
    """


    if model.endswith('.tflite'):
        tflite_path = fullpath(model)
        if not os.path.exists(tflite_path):
            cli.abort(msg=f'\n\n*** .tflite model file not found: {model}\n\n')

    else:
        try:
            mltk_model = load_mltk_model(model, print_not_found_err=True)
        except Exception as e:
            cli.abort(msg=f'\n\nFailed to load MltkModel, err: {e}\n\n')

        try:
            tflite_path = mltk_model.tflite_archive_path
        except Exception as e:
            cli.handle_exception(f'Failed to get .tflite from {mltk_model.archive_path}', e)

    old_generation_details = None
    generation_args_path = f'{os.path.dirname(output)}/generated_model_details.json'
    generation_details = dict(
        tflite_path=fullpath(tflite_path),
        tflite_hash=hash_file(tflite_path),
        output=fullpath(output)
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


    bin2header(
        input=tflite_path, 
        output_path=output, 
        var_name=variable_name,
        length_var_name=length_variable_name,
        attributes=variable_attributes,
    )
   
    with open(generation_args_path, 'w') as f:
        json.dump(generation_details, f, indent=3)

    print(f'Generated {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a model header file from a MLTK model or .tflite')
    parser.add_argument('model', help='Name of MLTK model or path to .tflite')
    parser.add_argument('--output', default='generated_model.tflite.h', help='Path to generated output header')
    parser.add_argument('--name', default='MODEL_DATA', help='Name of C array')
    parser.add_argument('--length_name', default='MODEL_DATA_LENGTH', help='Name of C variable to hold length of data in bytes')
    parser.add_argument('--attributes', default=None, help='Attributes to prepend to C array variable')

    args = parser.parse_args()
    try:
        generate_model_header(
            model=args.model,
            output=args.output, 
            variable_name=args.name,
            variable_attributes=args.attributes,
            length_variable_name=args.length_name
        )
    except Exception as _ex:
        cli.handle_exception('Failed to generate model header', _ex)