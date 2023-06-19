"""Utilities for invoking CMake

See the source code on Github: `mltk/utils/cmake.py <https://github.com/siliconlabs/mltk/blob/master/mltk/utils/cmake.py>`_
"""

from typing import List, Union, Dict
import sys
import os
import logging
import re
from cmake import CMAKE_BIN_DIR
CMAKE_BIN_DIR = CMAKE_BIN_DIR.replace('\\', '/')

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
    target:str=None,
    mltk_target:str=None,
    additional_variables:Union[List[str],Dict[str,str]]=None,
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
    use_user_options:bool=False,
    config_only:bool=False,
    build_only:bool=False
) -> str:
    """Build an MLTK CMake target

    Args:
        target: Name of CMake target to build
        mltk_target: Name of MLTK_TARGET, if omitted use target
        additional_variables: List or dictionary of additional CMake variables to add to the build command
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
        config_only: Only configure the CMake project, do not build it
        build_only: Only build the target, do not configure it first. In this case, the project must have been previosly configured
    Returns:
        The path to the build directory
    """

    logger = logger or DummyLogger()
    make_filelike(logger)

    mltk_target = mltk_target or target
    additional_variables = parse_variables(additional_variables)

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
        raise ValueError(f'Unsupported platform {platform}, supported platforms are: {support_platforms}')
    toolchain_file = PLATFORM_TOOLCHAIN_MAPPING[platform]

    python_dir = os.path.dirname(os.path.dirname(sys.executable)).replace('\\', '/')
    cmake_vars = parse_variables([
        #'CMAKE_OBJECT_PATH_MAX:STRING=1024',
        f'CMAKE_TOOLCHAIN_FILE:FILEPATH={MLTK_ROOT_DIR}/cpp/tools/toolchains/{toolchain_file}',
        f'CMAKE_BUILD_TYPE:STRING={build_type}',
        f'MLTK_PYTHON_VENV_DIR:FILEPATH={python_dir}',
        f'MLTK_PLATFORM_NAME:STRING={platform}',
        f'MLTK_TARGET:STRING={mltk_target}'
    ])
    if not use_user_options:
        cmake_vars['MLTK_NO_USER_OPTIONS'] = 'ON'
    if verbose:
        cmake_vars['MLTK_CMAKE_LOG_LEVEL:STRING'] = 'debug'
    if accelerator:
        cmake_vars['TFLITE_MICRO_ACCELERATOR:STRING'] = accelerator.lower()

    for key, value in additional_variables.items():
        value = str(value)
        # These args do not go through the shell parser
        # so they do not need to be wrapped with double-quotes
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        cmake_vars[key] = value

    cmake_vars_args = []
    for key, value in cmake_vars.items():
        cmake_vars_args.append(f'-D{key}={value}')

    if not build_only:
        cmd = [
            f'{CMAKE_BIN_DIR}/cmake',
            '--no-warn-unused-cli',
            '-Wno-dev',
            *cmake_vars_args,
            f'-S{source_dir}',
            f'-B{build_dir}',
            '-G Ninja'
        ]
        cmd_str = " ".join(cmd)
        logger.info(f'Configuring CMake project:\n{cmd_str}')
        retcode, _ = run_shell_cmd(
            cmd=cmd,
            outfile=logger,
        )
        if retcode != 0:
            raise RuntimeError(f'Failed to configure CMake project using command:\n{cmd_str}')


    cmd =[
         f'{CMAKE_BIN_DIR}/cmake',
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

    if not config_only:
        cmd_str = " ".join(cmd)
        logger.info(f'Building CMake project:\n{cmd_str}')
        retcode, _ = run_shell_cmd(
            cmd=cmd,
            outfile=logger
        )
        if retcode != 0:
            raise RuntimeError(f'Failed to build CMake project using command:\n{cmd_str}')

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

    cmd = [f'{CMAKE_BIN_DIR}/cmake']
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

    if build_subdir is False:
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



def parse_variables(
    cmake_variables:Union[List[str],Dict[str,str]]
) -> Dict[str,str]:
    """Convert a list or dictionary of CMake variables into a dictionary"""
    if not cmake_variables:
        return {}

    if isinstance(cmake_variables, (list,str)):
        var_re = re.compile(r'(.+)\s*=\s*(.*)')
        cmake_variables = as_list(cmake_variables)
        retval = {}
        for v in cmake_variables:
            if not v:
                continue

            match = var_re.match(v)
            if not match:
                raise ValueError(f'Invalid CMake variable: {v}, must be of form: <name>=<value>')

            key = match.group(1)
            value = match.group(2)

            if key.startswith('-D'):
                key = key[2:]
            retval[key] = value

        return retval
    else:
        return cmake_variables
