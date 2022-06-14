import os
import sys
import stat
import re
import multiprocessing
import pytest 

from mltk import MLTK_ROOT_DIR
from mltk.utils.test_helper import get_logger, run_mltk_command
from mltk.utils.shell_cmd import run_shell_cmd 
from mltk.utils.archive_downloader import download_verify_extract
from mltk.utils.path import create_tempdir, clean_directory, recursive_listdir, get_user_setting
from mltk.utils.python import import_module_at_path


#SPECIFIC_BOARD = 'brd4001a,brd4186c'

if os.name == 'nt':
    SIMPLICITY_STUDIO_PATH = get_user_setting('simplicity_studio_path') or 'C:/SiliconLabs/SimplicityStudio/v5'
    SIMPLICITY_STUDIO_MYSYS_BIN_PATH = f'{SIMPLICITY_STUDIO_PATH}/support/common/build/msys/1.0/bin'

    if not os.path.exists(SIMPLICITY_STUDIO_PATH):
        raise RuntimeWarning(
            'Simplicity Studio not found.\n'
            'You may download and install Simplicity Studio from here:\n'
            'https://www.silabs.com/developers/simplicity-studio\n'
            'Alternatively, create/update the file: ~/.mltk/user_settings.yaml\n'
            'and add the following:\n'
            'simplicity_studio_path: <path to Simplicity Studio>\n'
            'where <path to Simplicity Studio> points to the Simplicity Studio directory'
        )
else:
    SIMPLICITY_STUDIO_PATH = get_user_setting('simplicity_studio_path')
    if not SIMPLICITY_STUDIO_PATH:
        raise RuntimeWarning(
            'Simplicity Studio not found.\n'
            'Create/update the file: ~/.mltk/user_settings.yaml\n'
            'and add the following:\n\n'
            'simplicity_studio_path: <path to Simplicity Studio>\n\n'
            'where <path to Simplicity Studio> points to the extracted directory of the archive downloaded from:\n'
            'https://www.silabs.com/developers/simplicity-studio'
        )





build_params = []

ALL_BOARDS = [ 
    'brd2601b',
    'brd2204a',
    'brd4166a',
    'brd4001a,brd4186c'
]



def _add_build_params(app_name, boards):
    for b in boards:
        if 'SPECIFIC_BOARD' in globals() and b != globals()['SPECIFIC_BOARD']:
            continue
        build_params.append((app_name, b))

_add_build_params('hello_world', ALL_BOARDS)
_add_build_params('model_profiler', ALL_BOARDS)
_add_build_params('audio_classifier', ['brd2601b', 'brd2204a', 'brd4166a'])
_add_build_params('image_classifier', ALL_BOARDS)
_add_build_params('fingerprint_authenticator', ALL_BOARDS)




generate_slcp_logger = get_logger('simplicity_studio_tests', console=False)

gsdk_dir = None
slc_cli = None
gcc_dir = None
jre_bin_dir = None 


@pytest.mark.dependency()
def test_get_slc_cli() -> str:
    global gsdk_dir, slc_cli, gcc_dir

    download_arm_toolchain_module = import_module_at_path(f'{MLTK_ROOT_DIR}/cpp/tools/toolchains/gcc/arm/download_toolchain.py')
    gcc_dir = download_arm_toolchain_module.download_arm_toolchain(return_path=True)
    generate_slcp_logger.info(f'ARM GCC dir: {gcc_dir}')

    generate_slcp_logger.info('Installing the MLTK GSDK extension ...')
    run_mltk_command('build', 'gsdk_mltk_extension', '--no-show', logger=generate_slcp_logger)

    gsdk_dir = _get_gsdk_dir()
    slc_cli = _get_slc_exe()

    cmd = slc_cli + ['signature', 'trust', '-s', gsdk_dir]
    generate_slcp_logger.info(" ".join(cmd))
    retcode, retmsg = run_shell_cmd(cmd, outfile=generate_slcp_logger)
    if retcode != 0:
        raise RuntimeError(f'Failed to "trust" {gsdk_dir}\ncmd: {" ".join(cmd)}\nerr: {retmsg}')

    cmd = slc_cli + ['signature', 'trust', '-s', gsdk_dir, '-extpath', f'{gsdk_dir}/extension/mltk']
    generate_slcp_logger.info(" ".join(cmd))
    retcode, retmsg = run_shell_cmd(cmd, outfile=generate_slcp_logger)
    if retcode != 0:
        raise RuntimeError(f'Failed to "trust" MLTK extension\ncmd: {" ".join(cmd)}\nerr: {retmsg}')


@pytest.mark.dependency(depends=['test_get_slc_cli'])
@pytest.mark.parametrize(['app_name', 'board'], build_params)
def test_simplicity_studio_project(app_name, board):
    """Generate the SLCP project and build using make"""
    generate_slcp_logger.info('*' * 100)
    generate_slcp_logger.info(f'Generating/building {app_name} for {board}')
    app_dir = f'{MLTK_ROOT_DIR}/cpp/shared/apps/{app_name}'
    build_dir = create_tempdir(f'utest/simplicity_studio/{app_name}-{board.replace(",", "_")}')
    clean_directory(build_dir)
    
    cmd = slc_cli + ['generate', 
        '-s', gsdk_dir, 
        '--require-clean-project',
        '--new-project',
        '--copy-sources',
        '-d', build_dir,
        '--with', board,
        '-p', f'{app_dir}/mltk_{app_name}.slcp'
    ]
    generate_slcp_logger.info(" ".join(cmd))

    env = os.environ
    env['JAVA11_HOME'] = jre_bin_dir
    env['PATH'] = env['PATH'].replace(os.path.dirname(sys.executable + os.pathsep), '') # Clear the venv bin dir as that seems to be causing errors in the slc cli

    retcode, retmsg = run_shell_cmd(cmd, outfile=generate_slcp_logger, env=env)
    if retcode != 0:
        raise RuntimeError(f'Failed generate Simplicity Studio project for {app_name} app\ncmd: {" ".join(cmd)}\nerr: {retmsg}')

    env = os.environ
    if os.name == 'nt':
        os.environ["PATH"] = os.path.normpath(SIMPLICITY_STUDIO_MYSYS_BIN_PATH) + os.pathsep + os.environ["PATH"]

    env['ARM_GCC_DIR'] = gcc_dir

    cmd = ['make', '-f', f'mltk_{app_name}.Makefile', '-j', str(multiprocessing.cpu_count())]
    generate_slcp_logger.info(" ".join(cmd))
    retcode, retmsg = run_shell_cmd(cmd, outfile=generate_slcp_logger, env=env, cwd=build_dir)
    if retcode != 0:
        raise RuntimeError(f'Failed build Simplicity Studio project for {app_name} app\ncmd: {" ".join(cmd)}\nerr: {retmsg}')



def _get_slc_exe() -> str:
    """Download the slc_cli stand-alone utility and return the path to the executable"""
    WINDOWS_URL = 'https://www.silabs.com/documents/login/software/slc_cli_windows.zip'
    LINUX_URL = 'https://www.silabs.com/documents/login/software/slc_cli_linux.zip'

    url = WINDOWS_URL if os.name == 'nt' else LINUX_URL

    slc_dir = download_verify_extract( 
        url=url,
        dest_subdir='tools/slc_cli',
        remove_root_dir=True,
        logger=generate_slcp_logger
    )

    venv_dir = f'{slc_dir}/.venv'
    retcode, retmsg = run_shell_cmd([sys.executable, '-m', 'venv', venv_dir])
    if retcode != 0:
        raise RuntimeError(f'Failed to create venv at: {venv_dir}, err: {retmsg}')

    if os.name == 'nt':
        python_venv_exe = f'{venv_dir}/Scripts/python.exe'
    else:
        python_venv_exe = f'{venv_dir}/bin/python3'

    retcode, retmsg = run_shell_cmd([python_venv_exe, '-m', 'pip', 'install', '-r', f'{slc_dir}/requirements.txt'])
    if retcode != 0:
        raise RuntimeError(f'Failed to install slc requirements.txt, err: {retmsg}')


    exe_path = [python_venv_exe, f'{slc_dir}/slc']

    if os.name != 'nt':
        p = f'{slc_dir}/bin/slc-cli/slc-cli'
        mode = os.stat(p).st_mode
        mode |= stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
        os.chmod(p, mode)

    _get_java_bin_dir()

    retcode, retmsg = run_shell_cmd(exe_path + ['--help'])
    if retcode != 0:
        raise RuntimeError(f'Failed to verify slc executable: {slc_dir}/slc, err: {retmsg}')

    generate_slcp_logger.info(f'slc CLI path: {exe_path}')

    return exe_path


def _get_gsdk_dir() -> str:
    """Get the path to the GSDK repo downloaded by the MLTK build scripts 
    """
    gsdk_mltk_dir = f'{MLTK_ROOT_DIR}/cpp/shared/gecko_sdk'
    cmakeliststxt_path = f'{gsdk_mltk_dir}/CMakeLists.txt'

    cache_version_re = re.compile(r'\s*CACHE_VERSION\s+([\w\.]+)\s*.*')

    sdk_version = None 
    git_tag = None
    with open(cmakeliststxt_path, 'r') as f:
        for line in f:
            match = cache_version_re.match(line)
            if match:
                sdk_version = match.group(1)
                break
        
    if not sdk_version:
        raise RuntimeError(f'Failed to parse {cmakeliststxt_path} for GSDK version')


    gsdk_dir = f'{gsdk_mltk_dir}/{sdk_version}'
    generate_slcp_logger.info(f'GSDK directory: {gsdk_dir}')

    return gsdk_dir


def _get_java_bin_dir() -> str:
    """Return the JRE bin directory used by Simplicity Studio"""
    global jre_bin_dir
    found = False
    java_exe = 'java'
    if os.name == 'nt':
        java_exe += '.exe'

    def _is_java(p:str) -> bool:
        nonlocal found
        if not found and p.endswith(java_exe):
            found = True 
            return True 
        return False

    if os.name == 'nt':
        search_dir = f'{SIMPLICITY_STUDIO_PATH}/features'
    else:
        search_dir = f'{SIMPLICITY_STUDIO_PATH}/jre/bin'

    dlist = recursive_listdir(
        search_dir,
        regex=_is_java
    )

    if not dlist:
        raise RuntimeError(f'Failed to find JRE in {SIMPLICITY_STUDIO_PATH}')

    jre_bin_dir =  os.path.dirname(dlist[0])
    generate_slcp_logger.info(f'JRE bin dir: {jre_bin_dir}')

