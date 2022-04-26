import sys  
import os 


CURDIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(f'{CURDIR}/../../../../..')


from cpp.tools.utils.download_tool import download_tool 
from mltk.utils.path import create_user_dir



URL = 'https://github.com/brechtsanders/winlibs_mingw/releases/download/8.4.0-7.0.0-r1/mingw-w64-x86_64-8.4.0-7.0.0-r1.7z'
SHA1 = 'EECBC18AEB784A0A2350A16194177A4B25FD0904'
VERSION = '8.4.0'
SUBDIR = 'mingw64'


def download_win64_toolchain():
    curdir = os.path.dirname(os.path.abspath(__file__))

    dest_dir = create_user_dir(f'tools/toolchains/gcc/windows/{VERSION}')
    extracted_dir = f'{dest_dir}/{SUBDIR}'

    show_progress = '--noprogress' not in sys.argv

    # NOTE: If the tool already exists then this doesn't do anything
    retcode, retval = download_tool(
        url=URL, 
        name='GCC', 
        dest_dir=dest_dir, 
        extracted_dir=extracted_dir,
        file_hash=SHA1, 
        show_progress=show_progress,
        log_file=f'{curdir}/download.log',
        override_stdout_with_logger=True
    )

    sys.stdout.write(f'{retval};')
    sys.exit(retcode)


if __name__ == '__main__':
    download_win64_toolchain()