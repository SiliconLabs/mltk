import sys  
import os 
import traceback

CURDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.normpath(f'{CURDIR}/../../../../..'))




URL = 'https://github.com/brechtsanders/winlibs_mingw/releases/download/8.4.0-7.0.0-r1/mingw-w64-x86_64-8.4.0-7.0.0-r1.7z'
SHA1 = 'EECBC18AEB784A0A2350A16194177A4B25FD0904'
VERSION = '8.4.0'
SUBDIR = 'mingw64'


def download_win64_toolchain():
    retcode = 0
    log_file=f'{CURDIR}/download.log'
    show_progress = '--noprogress' not in sys.argv

    try:
        from cpp.tools.utils.download_tool import download_tool 
        from mltk.utils.path import create_user_dir

        dest_dir = create_user_dir(f'tools/toolchains/gcc/windows/{VERSION}')
        extracted_dir = f'{dest_dir}/{SUBDIR}'
    except Exception as e:
        retcode = -1
        with open(log_file, 'w') as f:
            f.write(traceback.format_exc() + '\n')
            f.write(f'Python path: {sys.executable}\n')
            f.write(f'Error while importing MLTK package, do you run the install_mltk.py script first?\nError details: {e}')  
            

    if retcode == 0:
        # NOTE: If the tool already exists then this doesn't do anything
        retcode, retval = download_tool(
            url=URL, 
            name='GCC', 
            dest_dir=dest_dir, 
            extracted_dir=extracted_dir,
            file_hash=SHA1, 
            show_progress=show_progress,
            log_file=log_file,
            override_stdout_with_logger=True
        )

    if retcode != 0:
        retval = log_file
        try:
            with open(log_file, 'r') as f:
                data = f.read()
            with open(log_file, 'w') as f:
                f.write(data.replace('\n', ';\n'))
        except:
            pass

    sys.stdout.write(f'{retval};')
    sys.exit(retcode)


if __name__ == '__main__':
    download_win64_toolchain()