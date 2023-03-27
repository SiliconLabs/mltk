import sys
import os
import yaml


CURDIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.normpath(f'{CURDIR}/../../../../..'))


from cpp.tools.utils.download_tool import download_tool
from mltk.utils.path import create_user_dir
from mltk.utils.system import get_current_os


def download_simulator_lib():
    current_os = get_current_os()
    with open(f'{CURDIR}/simulator_download_urls.yaml', 'r') as fp:
        download_urls = yaml.load(fp, Loader=yaml.SafeLoader)

    if current_os not in download_urls:
        raise Exception(f'OS {current_os} is not currently supported')

    download_info = download_urls[current_os]

    dest_dir = create_user_dir('accelerators/mvp/libs/simulator')
    sys.stdout.write(f'{dest_dir};')

    # NOTE: If the lib already exists then this doesn't do anything
    retcode, retval = download_tool(
        url=download_info['url'],
        name='MVP Simulator Library',
        dest_dir=dest_dir,
        file_hash=download_info.get('sha1', download_info.get('md5')),
        show_progress=False,
        log_file=f'{dest_dir}/download.log',
        download_details_fname='download_details.json',
        override_stdout_with_logger=True
    )

    if retcode == 0:
        sys.stdout.write(f'{dest_dir}/sl_mvp_simulator.a;')
    else:
        sys.stdout.write(f'{retval};')
    sys.exit(retcode)


if __name__ == '__main__':
    download_simulator_lib()