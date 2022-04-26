import os 
import sys 
import stat
import logging


from mltk.cli import  is_command_active
from mltk.utils.archive_downloader import download_verify_extract
from mltk.utils.system import get_current_os


DOWNLOAD_URLS = {
    'windows': dict( 
        url='https://github.com/SiliconLabs/mltk_assets/raw/master/tools/commander/Commander_win32_x64_1v12p0b1057.zip',
        subdir='v1.12',
        sha1='17C6D488A92979AA8F17C436D150A1F9A6F1694B'
    ),
    'linux': dict( 
        url='https://github.com/SiliconLabs/mltk_assets/raw/master/tools/commander/Commander_linux_x86_64_1v12p0b1057.tar.bz',
        subdir='v1.12',
        sha1='F45DC198D34546E17614FAA223EAB6B30E14ECD1'
    )
}


def download_commander(logger: logging.Logger = None) -> str:
    current_os = get_current_os()

    if current_os not in DOWNLOAD_URLS:
        raise Exception(f'OS {current_os} not supported')

    url_details = DOWNLOAD_URLS[current_os]
    url = url_details['url']

    # NOTE: This immediately returns if the util has already been downloaded
    dest_dir = download_verify_extract(
        url=url, 
        dest_subdir=f'tools/commander/{url_details["subdir"]}',
        show_progress=is_command_active(),
        file_hash=url_details['sha1'],
        file_hash_algorithm='sha1',
        remove_root_dir=True,
        logger=logger
    )

    exe_path = f'{dest_dir}/commander'

    if os.name == 'nt':
        exe_path += '.exe'
    else:
        # Set executable permissions on file
        mode = os.stat(exe_path).st_mode
        mode |= stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
        os.chmod(exe_path, mode)

    return exe_path 



if __name__ == '__main__':
    cmder_path = download_commander()
    sys.stdout.write(cmder_path)

