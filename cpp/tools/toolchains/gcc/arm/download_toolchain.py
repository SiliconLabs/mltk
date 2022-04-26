import sys  
import os 


CURDIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(f'{CURDIR}/../../../../..')


from cpp.tools.utils.download_tool import download_tool 
from mltk.utils.path import create_user_dir
from mltk.utils.system import get_current_os


# https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm/downloads
# WARNING: The latest ARM GCC is NOT working with Tensorflow
URLS = {
    'windows': dict(
        url='https://developer.arm.com/-/media/Files/downloads/gnu-rm/9-2020q2/gcc-arm-none-eabi-9-2020-q2-update-win32.zip',
        version='2020q2',
        md5='184b3397414485f224e7ba950989aab6',
        extract_subdir=''
    ),
    'linux': dict(
        url='https://developer.arm.com/-/media/Files/downloads/gnu-rm/9-2020q2/gcc-arm-none-eabi-9-2020-q2-update-x86_64-linux.tar.bz2',
        version='2020q2',
        md5='2b9eeccc33470f9d3cda26983b9d2dc6',
        extract_subdir='/gcc-arm-none-eabi-9-2020-q2-update'
    )
}



def download_arm_toolchain(return_path=False):
    curdir = os.path.dirname(os.path.abspath(__file__))
    current_os = get_current_os()

    if current_os not in URLS:
        raise Exception(f'OS {current_os} not supported')

    url_info = URLS[current_os]
    
    dest_dir = create_user_dir(f'tools/toolchains/gcc/arm/{url_info["version"]}')
    extracted_dir = dest_dir + url_info['extract_subdir']

    # NOTE: If the tool already exists then this doesn't do anything
    retcode, retval = download_tool(
        url=url_info['url'], 
        name='GCC', 
        dest_dir=dest_dir, 
        extracted_dir=extracted_dir,
        file_hash=url_info['md5'],
        show_progress=False,
        log_file=f'{curdir}/download.log',
        override_stdout_with_logger=True
    )

    if current_os == 'windows':
        ext = '.exe'
    else: 
        ext = ''

    if return_path:
        if retcode != 0:
            raise RuntimeError(f'Failed to download ARM GCC toolchain, err: {retval}')
        return retval
    else:
        sys.stdout.write(f'{retval};{ext};')
        sys.exit(retcode)


if __name__ == '__main__':
    download_arm_toolchain()