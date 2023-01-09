"""Utilities for invoking CMake

See the source code on Github: `mltk/utils/cmake.py <https://github.com/siliconlabs/mltk/blob/master/mltk/utils/cmake.py>`_
"""

from typing import List
import sys
import os
import logging
from cmake import CMAKE_BIN_DIR

from mltk import MLTK_ROOT_DIR
from .shell_cmd import run_shell_cmd
from .path import clean_directory, create_tempdir
from .system import get_current_os
from .logger import DummyLogger, make_filelike
from .python import as_list



PLATFORM_TOOLCHAIN_MAPPING = {
    'windows'  : 'gcc/windows/win64_toolchain.cmake',
    'linux'    : 'gcc/linux/linux_toolchain.cmake',
    'osx'      : 'gcc/osx/osx_toolchain.cmake',
    'brd2204'  : 'gcc/arm/arm_toolchain.cmake',
    'brd4186'  : 'gcc/arm/arm_toolchain.cmake',
    'brd2601'  : 'gcc/arm/arm_toolchain.cmake',
    'brd4166'  : 'gcc/arm/arm_toolchain.cmake',
}




def build_mltk_target(
    target:str,
    mltk_target:str=None,
    additional_variables:List[str]=None,
    debug:bool=False,
    clean:bool=False,
    source_dir:str=None,
    build_dir:str=None,
    build_subdir:str=None,
    platform:str=None,
    logger:logging.Logger=None,
    verbose:bool=False,
    jobs:int=None,
    accelerator:str=None,
    use_user_options:bool=False
) -> str:
    """Build an MLTK CMake target

    Args:
        target: Name of CMake target to build
        mltk_target: Name of MLTK_TARGET, if omitted use target
        additional_variables: List of additional CMake variables to add to the build command
        debug: If true then build with debugging (i.e. full symbols and no optimization)
        clean: Clean the build directory before building
        source_dir: Path to source directory, if omitted use mltk root directory
        build_dir: Path to build directory, if omitted use <temp dir>/mltk/build
        build_subdir: Name of sub build directory, if omitted use target name
        platform: Build platform, if omitted use current OS
        logger: Optional python logger
        verbose: Enable verbose logging while building
        accelerator: Name of accelerator to use for TFLITE_MICRO_ACCELERATOR CMake variable
        jobs: Number of parallel build jobs
        use_user_options: Use the user_options.cmake in the source directory. Default is to IGNORE user_options.cmake
    Returns:
        The path to the build directory
    """

    logger = logger or DummyLogger()
    make_filelike(logger)

    mltk_target = mltk_target or target
    additional_variables = as_list(additional_variables)

    if source_dir is None:
        source_dir = MLTK_ROOT_DIR
    source_dir = source_dir.replace('\\', '/')

    if debug:
        build_type = 'Debug'
    else:
        build_type = 'Release'

    platform = platform or get_current_os()

    build_dir = get_build_directory(
        platform=platform,
        target=target,
        debug=debug,
        build_dir=build_dir,
        build_subdir=build_subdir
    )

    if clean:
        clean_directory(build_dir)

    if platform not in PLATFORM_TOOLCHAIN_MAPPING:
        support_platforms = list(PLATFORM_TOOLCHAIN_MAPPING.keys())
        raise Exception(f'Unsupported platform {platform}, supported platforms are: {support_platforms}')
    toolchain_file = PLATFORM_TOOLCHAIN_MAPPING[platform]

    python_dir = os.path.dirname(os.path.dirname(sys.executable)).replace('\\', '/')
    cmd = [
        f'{CMAKE_BIN_DIR}/cmake'.replace('\\', '/'),
        '--no-warn-unused-cli',
        '-Wno-dev',
        '-DCMAKE_OBJECT_PATH_MAX:STRING=1024',
        f'-DCMAKE_TOOLCHAIN_FILE:FILEPATH={MLTK_ROOT_DIR}/cpp/tools/toolchains/{toolchain_file}',
        f'-DCMAKE_BUILD_TYPE:STRING={build_type}',
        f'-DMLTK_PYTHON_VENV_DIR:FILEPATH={python_dir}',
        f'-DMLTK_PLATFORM_NAME:STRING={platform}',
        f'-DMLTK_TARGET:STRING={mltk_target}'
    ]
    if not use_user_options:
        cmd.append('-DMLTK_NO_USER_OPTIONS=ON')
    if verbose:
        cmd.append('-DMLTK_CMAKE_LOG_LEVEL:STRING=debug')
    if accelerator:
        cmd.append(f'-DTFLITE_MICRO_ACCELERATOR:STRING={accelerator}')
    for v in additional_variables:
        cmd.append(f'-D{v}')

    cmd.extend([
        f'-S{source_dir}',
        f'-B{build_dir}',
        '-G Ninja'
    ])

    cmd_str = '\n'.join(cmd)
    logger.info('Invoking:\n' + cmd_str + '\n')
    retcode, _ = run_shell_cmd(
        cmd=cmd,
        outfile=logger,
    )
    if retcode != 0:
        raise Exception('Failed to configure CMake project')

    cmd = [
        f'{CMAKE_BIN_DIR}/cmake'.replace('\\', '/'),
        '--build', build_dir,
        '--config', build_type,
        '--target', target
    ]

    if verbose or jobs:
        cmd.append('--')

    if verbose:
        cmd.extend(['-d', 'keeprsp'])
        cmd.append('-v')

    if jobs:
        cmd.extend(['-j', str(jobs)])

    cmd_str = '\n'.join(cmd)
    logger.info(f'Invoking {cmd_str}')
    retcode, _ = run_shell_cmd(
        cmd=cmd,
        outfile=logger,
    )
    if retcode != 0:
        raise RuntimeError('Failed to build CMake project')

    return build_dir


def invoke_mltk_target(
    target:str,
    build_target:str=None,
    debug:bool=False,
    build_dir:str=None,
    build_subdir:str=None,
    platform:str=None,
    logger:logging.Logger=None,
    verbose:bool=False,
) -> str:
    """Invoke an MLTK CMake target"""

    logger = logger or DummyLogger()
    make_filelike(logger)

    platform = platform or get_current_os()

    build_dir = get_build_directory(
        platform=platform,
        target=build_target,
        debug=debug,
        build_dir=build_dir,
        build_subdir=build_subdir
    )

    cmd = [f'{CMAKE_BIN_DIR}/cmake'.replace('\\', '/')]
    cmd.extend(['--build', build_dir])
    cmd.extend(['--target', target])
    if verbose or (logger is not None and hasattr(logger, 'verbose') and logger.verbose):
        cmd.append('-v')

    logger.info(f'Invoking {" ".join(cmd)}')
    retcode, retval = run_shell_cmd(
        cmd=cmd,
        outfile=logger,
    )
    if retcode != 0:
        raise RuntimeError('Failed to invoke CMake target')
    return retval

def get_build_directory(
    platform:str=None,
    target:str=None,
    debug:bool=False,
    build_dir:str=None,
    build_subdir:str=None,
) -> str:
    if build_dir is None:
        build_dir = create_tempdir('build')
    build_dir = build_dir.replace('\\', '/')

    if build_subdir == False:
        return build_dir

    platform = platform or get_current_os()
    platform = platform.lower()

    if debug:
        build_type = 'Debug'
    else:
        build_type = 'Release'

    build_subdir = build_subdir or target

    build_dir += f'/{build_subdir}/{platform}/{build_type}'

    return build_dir
