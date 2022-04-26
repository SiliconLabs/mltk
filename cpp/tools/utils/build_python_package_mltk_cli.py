from typing import List
import sys
import os
import re
import shutil
import logging
import time
import copy
import typer
from mltk import cli

from mltk import __version__ as mltk_version
from mltk import MLTK_ROOT_DIR
from mltk.utils.shell_cmd import run_shell_cmd
from mltk.utils.path import (create_tempdir, remove_directory, recursive_listdir, get_user_setting, fullpath)
from mltk.utils.python import install_pip_package
from mltk.utils.system import is_linux


@cli.build_cli.command('python_package')
def build_python_package_command(
    python_exe: str = typer.Option(None, '--python', '-p', 
        help='Path to Python executable or Python command found on PATH. If omitted, use current Python'
    ),
    clean_venv: bool = typer.Option(True, 
        help='Clean the Python virtual environment before building'
    ),
    install: bool = typer.Option(True,
        help='Install the local repo into the venv, e.g.: pip install -e .'
    ),
    build: bool = typer.Option(True,
        help='Build the MLTK wheel'
    ),
    utests: str = typer.Option('api',
        help='''\b
Run the MLTK unit tests against the built Python wheel in a new venv, BEFORE releasing to pypi.org. 
This should be a comma-separated list of unit tests to run. See "mltk utest --help" for more details. 
Set as "none" to skip tests'''
    ),
    release_test: bool = typer.Option(False,
        help='Release the built wheel to test.pypi.org'
    ),
    release_public: bool = typer.Option(False,
        help='Release the built wheel to pypi.org'
    ),
    release_utests: str = typer.Option('api',
        help='''\b
Run the MLTK unit tests against the released package on pypi.org.
This should be a comma-separated list of unit tests to run. See "mltk utest --help" for more details. 
Set as "none" to skip tests'''
    ),
    release_all: bool = typer.Option(False, '--all',
            help='''\b
Release for all supported Python versions.
~/.mltk/user_settings.yaml must have a field python_paths: which contains list of Python executable paths, e.g.:
 python_paths:
  - C:/Python37/python.exe
  - C:/Python38/python.exe
  - C:/Python39/python.exe
'''
    ),
    pip_packages: str = typer.Option(None,
        help="""Force specific PIP packages during the unit tests. Each package should be delimited by a pipe character |
        \b
        e.g.: --pip-packages "tensorflow==2.4.*|tensorflow-probability==0.12.0"
        """
    ),
):
    """Build a MLTK wheel for a specific Python version and optionally release to pypi.org

    \b
    To release the built wheel to https://test.pypi.org, add the --release-test option.
    To use this option, first update/create the file ~/.mltk/user_settings.yaml,
    and add: 
        test_pypi_token: <token>
    where <token> is your test.pypi.org "API Token"
    \b
    Once released, the wheel may be installed via:
        pip install --extra-index-url https://test.pypi.org/simple silabs-mltk
    \b
    To release the built wheel to https://pypi.org, add the --release-public option.
    To use this option, first update/create the file ~/.mltk/user_settings.yaml,
    and add:
          pypi_token: <token>
    where <token> is your pypi.org "API Token"
    \b
    Once released, the wheel may be installed via:
        pip install silabs-mltk

    NOTE: Before releasing, the __version__ in <mltk repo>/mltk/__init__.py must be incremented.

    This effectively runs the commands:

    \b
    if --clean-venv:
        rm -rf  temp/mltk/python_venvs/<python version>
    python -m venv temp/mltk/python_venvs/<python version>
    \b
    if --install:
        export MLTK_NO_BUILD_WRAPPERS=1
        <venv pip> install -e .
    \b
    if --build:
        <venv python> setup.py bdist_wheel
    \b
    if --utests:
        rm -rf  temp/mltk/python_venvs/tests/<python version>
        python -m venv temp/mltk/python_venvs/tests/<python version>
        <venv python> install <built wheel>
        mltk utest all
    \b
    if --release-test:
        twine upload --repository testpypi dist/*
        if --release-utests:
            <venv pip> install --extra-index-url https://test.pypi.org/simple silabs-mltk
            mltk utest all
    \b
    if --release-public:
        twine upload dist/*
        if --release-utests:
            <venv pip> install silabs-mltk
            mltk utest all

    HINT: Add the --all option to release for all Python versions at once
    """

    logger = cli.get_logger(verbose=True)


    if release_all or release_test or release_public:
        install_pip_package('twine', logger=logger)

        retcode, dst_mltk_origin_url = run_shell_cmd(['git', 'config', '--get', 'remote.origin.url'], cwd=MLTK_ROOT_DIR)
        if retcode != 0:
            cli.abort(msg=f'Failed to get remote.origin.url from {MLTK_ROOT_DIR}, err: {dst_mltk_origin_url}')

        public_mltk_dir = get_user_setting('public_mltk_dir')
        if public_mltk_dir is None:
            cli.abort(msg='Must specify "public_mltk_dir: <github mltk repo dir>" in ~/.mltk/user_settings.yaml which points to directory of the cloned mltk repo on github')

        public_mltk_dir = fullpath(public_mltk_dir)
        current_mltk_dir = fullpath(MLTK_ROOT_DIR)
        if public_mltk_dir != current_mltk_dir:
            cli.abort(
                msg=f'~/.mltk/user_settings.yaml:public_mltk_dir={public_mltk_dir}\n' \
                f'does not match the current MLTK_ROOT_DIR={current_mltk_dir}\n' \
                'You must only release the silabs-mltk package from the public github repo!'
            )


    if release_all:
        python_paths:List[str] = get_user_setting('python_paths')
        if not python_paths:
            cli.abort(msg='~/.mltk/user_settings.yaml must have a field python_paths: which contains list of Python executable paths, e.g.:\n' + \
                'python_paths:\n' + \
                '  - C:/Python37/python.exe\n' + \
                '  - C:/Python38/python.exe\n' + \
                '  - C:/Python39/python.exe\n'
            )

        for python_path in python_paths: # pylint: disable=not-an-iterable
            cmd = copy.deepcopy(sys.argv)
            cmd.remove('--all')
            cmd.extend(['--python', python_path])
            retcode, _ = run_shell_cmd(cmd, outfile=logger)
            if retcode != 0:
                cli.abort(code=retcode, msg=f'Failed to release wheel for {python_path}')


    #######################################
    # Determine the Python version

    if not python_exe:
        python_exe = sys.executable

    logger.info(f'Build MLTK wheel using {python_exe} ...')

    retcode, retmsg = run_shell_cmd([python_exe, '--version'], outfile=logger)
    if retcode != 0:
        cli.abort(msg=f'Failed to get Python version, err: {retmsg}\nEnsure the given Python executable is valid')

    match = re.match(r'.*\s(\d+).(\d+).(\d+)', retmsg.strip())
    if not match:
        cli.abort(msg=f'Failed to get Python version from {retmsg}')

    python_version_major = match.group(1)
    python_version_minor = match.group(2)
    python_version = f'{python_version_major}.{python_version_minor}'

    ##########################################
    # Create the Python virtual environment

    python_venv_dir = create_tempdir(f'release/python_venvs/{python_version}')
    setup_py_path = f'{MLTK_ROOT_DIR}/setup.py'

    if clean_venv:
        logger.info(f'Cleaning {python_venv_dir} ...')
        remove_directory(python_venv_dir)
    os.makedirs(python_venv_dir, exist_ok=True)

    logger.info(f'Creating Python v{python_version} virtual environment at {python_venv_dir}')
    retcode, retmsg = run_shell_cmd([python_exe, '-m', 'venv', python_venv_dir], outfile=logger)
    if retcode != 0:
        additional_msg = ''
        if 'ensurepip' in retmsg:
            additional_msg += '\n\nTry running the following command first:\n'
            additional_msg += f'sudo apt-get install python{python_version}-venv\n\n'
        
        cli.abort(msg=f'Failed to create Python venv, err: {retmsg}{additional_msg}')

    if os.name == 'nt':
        python_venv_exe = f'{python_venv_dir}/Scripts/python.exe'  
    else:
        python_venv_exe = f'{python_venv_dir}/bin/python3'

    # Work around install error
    run_shell_cmd([python_venv_exe, '-m', 'pip', 'install', '-U', 'certifi'], outfile=logger)


    if release_test:
        test_pypi_token = get_user_setting('test_pypi_token')
        if test_pypi_token is None:
            cli.abort(
                msg='When using the --release-test option, the file ~/.mltk/user_settings.yaml must have the line: "test_pypi_token: <token>"'
                'which points the the test.pypi.org API token'
            )
        _check_pip_version(python_venv_exe, python_version, use_test_pypi=True, logger=logger)
    
    if release_public:
        pypi_token = get_user_setting('pypi_token')
        if pypi_token is None:
            cli.abort(
                msg='When using the --release-public option, the file ~/.mltk/user_settings.yaml must have the line: "pypi_token: <token>"'
                'which points the the pypi.org API token'
            )
        _check_pip_version(python_venv_exe, python_version, use_test_pypi=False, logger=logger)


    #################################
    # Install the local MLTK repo into the venv

    if install:
        logger.info('#' * 100)
        logger.info(f'Installing local MLTK package into {python_venv_dir} ...')
        env = os.environ.copy()
        env['MLTK_NO_BUILD_WRAPPERS'] = '1'
        cmd = [python_venv_exe, '-m', 'pip', 'install']
        if clean_venv:
            cmd.append('--force-reinstall')
        cmd.extend(['-e', MLTK_ROOT_DIR])
        retcode, retmsg = run_shell_cmd(cmd, env=env, outfile=logger, cwd=MLTK_ROOT_DIR)
        if retcode != 0:
            additional_msg = ''
            if 'includes non-existent path' in retmsg:
                additional_msg += '\n\nTry running the following command first:\n'
                additional_msg += f'sudo apt-get install python{python_version}-dev\n\n'
            
            cli.abort(msg=f'Failed to install MLTK Python package, err: {retmsg}{additional_msg}')


    #################################
    # Build the MLTK wheel

    if build:
        logger.info('#' * 100)
        logger.info(f'Building the MLTK Python wheel for Python {python_version} ...')

        remove_directory(f'{MLTK_ROOT_DIR}/dist')

        retcode, retmsg = run_shell_cmd(
            [python_venv_exe, setup_py_path, 'bdist_wheel'], 
            outfile=logger, 
            cwd=MLTK_ROOT_DIR
        )
        if retcode != 0:
            cli.abort(msg=f'Failed to build MLTK Python wheel, err: {retmsg}')


    #################################
    # Get the path to the built wheel

    mltk_version_regex = mltk_version.replace('.', '\\.')
    wheel_paths = recursive_listdir(
        base_dir=f'{MLTK_ROOT_DIR}/dist',
        regex=f'.*/silabs_mltk-{mltk_version_regex}-\\d+-cp{python_version_major}{python_version_minor}-.*' + '\\.whl'
    )
    if not wheel_paths:
        cli.abort(msg=f'Failed to find built .whl file in {MLTK_ROOT_DIR}/dist')

    wheel_path = wheel_paths[0].replace('\\', '/')

    if is_linux():
        # FIXME: This is a hack to enable the built wheel to
        # be uploaded to pypi.org.
        # Technically, the wheel should be built in Docker container
        # that allows for building actual "manylinux" wheels
        # More details here:
        # https://github.com/pypa/manylinux
        #
        # NOTE: The build scripts statically link most C libs
        #  and force GCC 2.17, see:
        # <mltk repo>/cpp/shared/platforms/linux/CMakeLists.txt
        #
        # The built wheel has been verified to work on Google Colab
        # and AWS lambda Docker
        new_path = wheel_path.replace('linux_x86_64', 'manylinux2014_x86_64')
        shutil.copy(wheel_path, new_path)
        wheel_path = new_path

    logger.info('\n\n\n***')
    logger.info(f'Built wheel path: {wheel_path}' + '\n\n\n')


    ##########################################
    # Run the MLTK unit tests
    _run_unit_tests( 
        utests=utests,
        pip_args=[wheel_path],
        logger=logger,
        python_exe=python_exe,
        python_version=python_version,
        pip_packages=pip_packages
    )


    ################################
    # Upload wheel to https://test.pypi.org
    if release_test:
        logger.info('#' * 100)
        logger.info(f'Uploading {wheel_path} to https://test.pypi.org ...')
        retcode, retmsg = run_shell_cmd(
            [sys.executable, '-m', 'twine', 'upload', '--repository', 'testpypi', '-u', '__token__', '-p', test_pypi_token, wheel_path], 
            outfile=logger
        )
        if retcode != 0:
            cli.abort(msg=f'Failed to run upload to https://test.pypi.org, err: {retmsg}')

        _run_unit_tests( 
            pip_args=['--extra-index-url', 'https://test.pypi.org/simple/', f'silabs-mltk=={mltk_version}'],
            logger=logger,
            python_exe=python_exe,
            python_version=python_version,
            pip_packages=pip_packages,
            retry=True,
            utests=release_utests
        )

    ################################
    # Upload wheel to https://pypi.org
    if release_public:
        logger.info('#' * 100)
        logger.info(f'Uploading {wheel_path} to https://pypi.org ...')
        retcode, retmsg = run_shell_cmd(
            [sys.executable, '-m', 'twine', 'upload', '-u', '__token__', '-p', pypi_token, wheel_path], 
            outfile=logger
        )
        if retcode != 0:
            cli.abort(msg=f'Failed to run upload to https://pypi.org, err: {retmsg}')
        
        _run_unit_tests( 
            pip_args=[f'silabs-mltk=={mltk_version}'],
            logger=logger,
            python_exe=python_exe,
            python_version=python_version,
            pip_packages=pip_packages,
            retry=True,
            utests=release_utests
        )

    logger.info('Done')



def _run_unit_tests(
    utests:str,
    logger:logging.Logger, 
    python_version:str,
    python_exe:str,
    pip_args:List[str],
    pip_packages:str,
    retry:bool = False,
):
    logger.info('#' * 100)
    logger.info('Installing built wheel in virtual environment ...')
    python_test_venv_dir = create_tempdir(f'release/python_venvs/tests/{python_version}')
    logger.info(f'Cleaning {python_test_venv_dir} ...')
    remove_directory(python_test_venv_dir)
    os.makedirs(python_test_venv_dir, exist_ok=True)
   
    logger.info(f'Creating Python v{python_version} virtual environment at {python_test_venv_dir}')
    retcode, retmsg = run_shell_cmd([python_exe, '-m', 'venv', python_test_venv_dir], outfile=logger)
    if retcode != 0:
        cli.abort(msg=f'Failed to create Python venv, err: {retmsg}')

    if os.name == 'nt':
        python_venv_exe = f'{python_test_venv_dir}/Scripts/python.exe'
    else:
        python_venv_exe = f'{python_test_venv_dir}/bin/python3'


    logger.info('#' * 100)
    pip_cmd = [python_venv_exe, '-m', 'pip', 'install', '--force-reinstall']
    pip_cmd.extend(pip_args)
    cmd_str = ' '.join(pip_cmd)
    logger.info(f'Run {pip_cmd}')
    for i in range(3):
        retcode, retmsg = run_shell_cmd(pip_cmd, outfile=logger)
        if retcode != 0:
            if i < 2 and retry:
                time.sleep(3)
                continue
            cli.abort(msg=f'PIP cmd failed, err: {retmsg}')
        break

    if pip_packages:
        logger.info('#' * 100)
        logger.info(f'Forcing PIP versions: {pip_packages}')
        cmd = [python_venv_exe, '-m', 'pip', 'install']
        for pkg in pip_packages.split('|'):
            cmd.append(pkg)
        retcode, retmsg = run_shell_cmd(cmd, outfile=logger)
        if retcode != 0:
            cli.abort(msg=f'Failed to force Tensorflow version, err: {retmsg}')


    if os.name == 'nt':
        mltk_exe = f'{python_test_venv_dir}/Scripts/mltk.exe'
    else:
        mltk_exe = f'{python_test_venv_dir}/bin/mltk'

    retcode, retmsg = run_shell_cmd([mltk_exe, '--help'], outfile=logger)
    if retcode != 0:
        cmd_str = f'{mltk_exe} --help'
        cli.abort(msg=f'Failed to run simple mltk cmd: {cmd_str}, err: {retmsg}')

    if utests:
        if utests.lower() in ('0', 'none', 'off', 'no'):
            utests = None

    if utests:
        logger.info('#' * 100)
        logger.info('Running MLTK unit tests ...')
        retcode, retmsg = run_shell_cmd([mltk_exe, 'utest', utests], outfile=logger, cwd=python_test_venv_dir)
        if retcode != 0:
            cli.abort(msg=f'Failed to run MLTK unit tests, err: {retmsg}')


def _check_pip_version(venv_python_exe, python_version, use_test_pypi, logger):
    logger.info('Checking if package version already exists on server ...')
    cmd = [venv_python_exe, '-m', 'pip', 'install']
    if use_test_pypi:
        cmd.extend(['-i', 'https://test.pypi.org/simple/'])
    cmd.append('silabs-mltk==')

    _, retmsg = run_shell_cmd(cmd)
   
    start_index = retmsg.find('(from versions:')
    if start_index == -1:
        return 
    retmsg = retmsg[start_index + len('(from versions:'):].strip()
    end_index = retmsg.find(')')
    if end_index == -1:
        return 
    retmsg = retmsg[:end_index]

    server_name = 'https://test.pypi.org' if use_test_pypi else 'https://pypi.org'
    available_versions = [x.strip() for x in retmsg.split(',')]
    logger.info(f'MLTK versions for Python v{python_version} found on {server_name}: {available_versions}')

    for v in available_versions:
        if v == mltk_version:
            cli.abort(
                msg=f'MLTK version: {mltk_version} for Python v{python_version} is already available on {server_name}\n'
                'Must update __version__ in <mltk repo>/mltk/__init__.py'
            )