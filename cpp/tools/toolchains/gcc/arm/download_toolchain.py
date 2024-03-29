import sys  
import os 
import traceback


CURDIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.normpath(f'{CURDIR}/../../../../..'))



# https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm/downloads
URLS = {
    'windows': dict(
        url='https://developer.arm.com/-/media/Files/downloads/gnu-rm/10.3-2021.10/gcc-arm-none-eabi-10.3-2021.10-win32.zip',
        version='2021.10',
        md5='2bc8f0c4c4659f8259c8176223eeafc1',
        extract_subdir='/gcc-arm-none-eabi-10.3-2021.10'
    ),
    'linux': dict(
        url='https://developer.arm.com/-/media/Files/downloads/gnu-rm/10.3-2021.10/gcc-arm-none-eabi-10.3-2021.10-x86_64-linux.tar.bz2',
        version='2021.10',
        md5='2383e4eb4ea23f248d33adc70dc3227e',
        extract_subdir='/gcc-arm-none-eabi-10.3-2021.10'
    )
}



def download_arm_toolchain(return_path=False):
    retcode = 0
    log_file=f'{CURDIR}/download.log'

    try:
        from cpp.tools.utils.download_tool import download_tool 
        from mltk.utils.path import create_user_dir
        from mltk.utils.system import get_current_os

        current_os = get_current_os()

        if current_os not in URLS:
            raise Exception(f'OS {current_os} not supported')

        url_info = URLS[current_os]
        
        dest_dir = create_user_dir(f'tools/toolchains/gcc/arm/{url_info["version"]}')
        extracted_dir = dest_dir + url_info['extract_subdir']
    except Exception as e:
        retcode = -1
        with open(log_file, 'w') as f:
            f.write(traceback.format_exc() + '\n')
            f.write(f'Python path: {sys.executable}\n')
            f.write(f'Error while importing MLTK package, do you run the install_mltk.py script first?\nError details: {e}')    
            

    if retcode == 0:
        # NOTE: If the tool already exists then this doesn't do anything
        retcode, retval = download_tool(
            url=url_info['url'], 
            name='GCC', 
            dest_dir=dest_dir, 
            extracted_dir=extracted_dir,
            file_hash=url_info['md5'],
            show_progress=False,
            log_file=log_file,
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
        if retcode != 0:
            retval = log_file
            try:
                with open(log_file, 'r') as f:
                    data = f.read()
                with open(log_file, 'w') as f:
                    f.write(data.replace('\n', ';\n'))
            except:
                pass

        sys.stdout.write(f'{retval};{ext};')
        sys.exit(retcode)


if __name__ == '__main__':
    download_arm_toolchain()