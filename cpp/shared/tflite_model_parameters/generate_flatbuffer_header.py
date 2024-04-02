import os
import sys 

from mltk.core.tflite_model_parameters import schema
from mltk.utils.shell_cmd import run_shell_cmd


curdir = os.path.dirname(os.path.abspath(__file__))


tflite_model_parameters_schema_dir = os.path.dirname(os.path.abspath(schema.__file__))

fbs_path = f'{tflite_model_parameters_schema_dir}/dictionary.fbs'


retcode, _ = run_shell_cmd(['flatc', '--version'])
if retcode == 0:
    flatc = 'flatc'
else:
    retcode, retmsg = run_shell_cmd([f'{curdir}/flatc', '--version'])
    if retcode == 0:
        flatc = f'{curdir}/flatc'
    else:
        print("  You must first install the flatbuffer compiler executable.")
        print("  You can either install it into the environment PATH")
        print("  OR you may download the executable from:")
        print("  https://github.com/google/flatbuffers/releases/tag/v23.5.26")
        print(f"  and extract to the directory: {curdir}")
        sys.exit(-1)

output_header_dir = f'{curdir}/tflite_model_parameters/schema'
output_header_path = f'{output_header_dir}/dictionary_generated.h'
retcode, retmsg = run_shell_cmd([flatc, '--cpp', '--scoped-enums', '-o', output_header_dir, fbs_path])
if retcode != 0:
    print(f'Failed to generate C++ flatbuffer header, err: {retmsg}')
    sys.exit(-1)


output_header_modified = ''
with open(output_header_path, 'r')  as fp:
    for line in fp:
        if line.strip().startswith('FLATBUFFERS_VERSION_REVISION == 8'):
            line = line.replace('8', '6') # Tensorflow-Lite Micro currently expects Flatbuffers 2.0.6
        
        if line.startswith('#include "flatbuffers/flatbuffers.h"'):
            output_header_modified += line 
            output_header_modified += '\n#undef FLATBUFFERS_FINAL_CLASS\n'
            output_header_modified += '#define FLATBUFFERS_FINAL_CLASS\n\n'
            output_header_modified += 'namespace mltk {\n'
            output_header_modified += 'namespace schema {\n\n'
            continue 

        if line.startswith('#endif  // FLATBUFFERS_GENERATED_DICTIONARY_H_'):
            output_header_modified += '} // namespace schema\n'
            output_header_modified += '} // namespace mltk\n\n'
            output_header_modified += line 
            continue

        if line.startswith('struct Entry FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {'):
            line = 'struct Entry : private flatbuffers::Table {'

        output_header_modified += line 



with open(output_header_path, 'w') as fp:
    fp.write(output_header_modified)
