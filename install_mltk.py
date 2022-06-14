import sys
import platform

_version = sys.version_info
_architecture_bits = platform.architecture()[0]
if _version[0] != 3 or _version[1] < 7 or _version[1] >= 10 or _architecture_bits != '64bit':
    sys.stdout.write(f'Cannot install MLTK, Python 64-bit, version 3.7, 3.8, or 3.9 is required (current version is: Python {_architecture_bits}  v{_version[0]}.{_version[1]})\n')
    sys.stdout
    sys.exit(-1)

import argparse 

import os
import re
import shutil 
import logging
import subprocess
import queue

from concurrent.futures import ThreadPoolExecutor




script_curdir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')


def main():
    logging.basicConfig(stream=sys.stdout, format='%(message)s', level='DEBUG')

    parser = argparse.ArgumentParser(description='Utility to install the mltk Python package for local development')
    parser.add_argument('--python', 
        help='Path to python executable used to install the mltk package. If omitted, use the current python executable', 
        default=None
    )
    parser.add_argument('--repo-path', 
        help='Path to mltk git repo. This is used by the --dev option. If omitted, use same directory as this script', 
        default=None
    )
    parser.add_argument('--no-verbose', 
        help='Disable verbose log messages', 
        default=False,
        action='store_true'
    )

    args = parser.parse_args()
    install_mltk_for_local_dev(
        python=args.python, 
        repo_path=args.repo_path,
        verbose=not args.no_verbose
    )



def install_mltk_for_local_dev(
    python:str=None, 
    repo_path:str=None,
    verbose:bool=False,
):
    """Install the mltk for local development
    
    Args:
        python: Path to python executable used to install the mltk package. If omitted, use the current python executable
        repo_path: Path to mltk git repo. If omitted, use same directory as this script
    """

    python = python or sys.executable
    repo_path = repo_path or script_curdir
    
    python = python.replace('\\', '/')
    repo_path = repo_path.replace('\\', '/')
    python_version = get_python_version(python)
    

    logging.info('Installing mltk for local development')
    logging.info(f'  Python: {python}')
    logging.info(f'  mltk git repo: {repo_path}')

    repo_setup_py_path = f'{repo_path}/setup.py'
    if not os.path.exists(repo_setup_py_path):
        logging.error('mltk git repo does not have a setup.py script. Is this a valid mltk repo?')
        sys.exit(-1)

    venv_dir = f'{repo_path}/.venv'
    logging.info(f'Creating Python virtual environment at {venv_dir} ...')
    try:
        issue_shell_command(python, '-m', 'venv', venv_dir)
    except Exception as e:
        err_msg = f'{e}'
        additional_msg = ''
        if 'ensurepip' in err_msg:
            additional_msg += '\n\nTry running the following commands first:\n'
            additional_msg += f'sudo apt-get -y install python{python_version}-venv python{python_version}-dev\n\n'
        raise Exception(f'{err_msg}{additional_msg}') #pylint: disable=raise-missing-from


    if os.name == 'nt':
        python_venv_exe = f'{venv_dir}/Scripts/python.exe'
    else:
        python_venv_exe = f'{venv_dir}/bin/python3'


    # Ensure the wheel package is installed
    logging.info('Installing the "wheel" Python package into the virtual environment')
    issue_shell_command(python_venv_exe, '-m', 'pip', 'install', 'wheel')

    logging.info(f'Installing MLTK into {venv_dir} ...')
    logging.info('(Please be patient, this may take awhile)')
    try:
        env = os.environ.copy()
        cmd = [python_venv_exe, '-m', 'pip', 'install']
        if verbose:
            env['MLTK_VERBOSE_INSTALL'] = '1'
            cmd.append('-v')
        if 'PYTHONHOME' in env:
            del env['PYTHONHOME']
        env['PATH'] = os.path.dirname(python_venv_exe) + os.pathsep + env['PATH']

        cmd.append('-e')

        issue_shell_command(*cmd, repo_path, env=env)
    except Exception as e:
        err_msg = f'{e}'
        additional_msg = ''
        if os.name != 'nt':
            additional_msg += '\n\nTry running the following commands first:\n'
            additional_msg += f'sudo apt-get -y install build-essential g++-8 gdb python{python_version}-dev libportaudio2 pulseaudio p7zip-full\n'
            additional_msg += '\n\n'
        raise Exception(f'{err_msg}{additional_msg}') #pylint: disable=raise-missing-from

    logging.info('Done\n\n')
    logging.info('The MLTK has successfully been installed!\n')
    logging.info('Issue the following command to activate the MLTK Python virtual environment:')
    if os.name == 'nt':
        if shutil.which('source'):
            logging.info(f'source {venv_dir}/Scripts/activate\n')
        else:
            logging.info(venv_dir.replace('/', '\\') + '\\Scripts\\activate.bat\n')
    else:
        logging.info(f'source {venv_dir}/bin/activate\n')
 


def get_python_version(python:str) -> str:
    if os.name == 'nt':
        python = python.replace('/', '\\')
    python_version_raw_str = subprocess.check_output([python, '--version'], text=True)
    match = re.match(r'.*\s(\d+).(\d+).(\d+)', python_version_raw_str.strip())
    if not match:
        logging.error(f'Failed to get Python version from {python_version_raw_str}')
        sys.exit(-1)

    return f'{match.group(1)}.{match.group(2)}'


def issue_shell_command(*args, env=None):
    cmd = [x for x in args]
    if os.name == 'nt':
        cmd[0] = cmd[0].replace('/', '\\')

    cmd_str = ' '.join(cmd)
    logging.info(cmd_str)
    try:
        p = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            text=True, # convert the shell output to a string (instead of bytes)
            close_fds=True,
            env=env
        )
    except Exception as e:
        logging.error(f'Failed to issue command: {cmd_str}, err: {e}')

    retval = ''
    for out_line, err_line in _read_popen_pipes(p):
        if out_line:
            retval += out_line
            logging.info(out_line.rstrip())
        if err_line:
            logging.info(err_line.rstrip())

    retcode = p.poll()
    if retcode != 0:
        sys.exit(retcode)

    return retval


def _enqueue_output(file, q):
    for line in iter(file.readline, ''):
        q.put(line)
    file.close()

def _read_popen_pipes(p):
    with ThreadPoolExecutor(2) as pool:
        q_stdout, q_stderr = queue.Queue(), queue.Queue()

        pool.submit(_enqueue_output, p.stdout, q_stdout)
        pool.submit(_enqueue_output, p.stderr, q_stderr)

        while True:
            if p.poll() is not None and q_stdout.empty() and q_stderr.empty():
                break

            out_line = err_line = ''

            try:
                out_line = q_stdout.get_nowait()
            except queue.Empty:
                pass
            try:
                err_line = q_stderr.get_nowait()
            except queue.Empty:
                pass

            yield (out_line, err_line)


if __name__ == '__main__':
    main()