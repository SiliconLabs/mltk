"""Shell command utility

See the source code on Github: `mltk/utils/shell_cmd.py <https://github.com/siliconlabs/mltk/blob/master/mltk/utils/shell_cmd.py>`_
"""

import os
import tempfile
import logging
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple,Union,Iterable,Callable,IO
import subprocess

from .signal_handler import SignalHandler



def run_shell_cmd(
    cmd: Union[str,Iterable],
    cwd:str=None,
    env:dict=None,
    outfile:IO=None,
    line_processor: Callable[[str],str]=None,
    logger:logging.Logger=None
) -> Tuple[int,str]:
    """Issue shell command

    Args:
        cmd: The shell command. This may be a string or a list of strings
        cwd: A path to a directory to change to before executing. If omitted then use the current working directory
        env: Dictionary of environment variables. If omitted then use current environment variables
        outfile:An opened, file-like object to write the shell command's output
        line_processor: Callback to be invoked for each line returned by shell command
        logger: Logger to dump shell command output
    Return:
        (retcode, retmsg)
    """
    si = None
    if os.name == 'nt':
        si = subprocess.STARTUPINFO()
        si.dwFlags = subprocess.SW_HIDE
        try:
            cmd[0] = cmd[0].replace('/', '\\')
        except:
            pass

    if isinstance(cmd, str):
        use_shell = True
    else:
        use_shell = False
        cmd = [str(x) for x in cmd]

    if logger is not None:
        cmd_str = ''
        if cwd:
            cmd_str += f'CWD:{cwd}, '
        if isinstance(cmd, (list,tuple)):
            cmd_str += ' '.join(cmd)
        else:
            cmd_str += ' ' + cmd
        logger.debug(cmd_str)

    process_line_by_line = line_processor is not None or outfile is not None

    out_pipe = subprocess.PIPE if process_line_by_line else tempfile.SpooledTemporaryFile()
    err_pipe = subprocess.PIPE if process_line_by_line else tempfile.SpooledTemporaryFile()
    try:
        p = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=out_pipe,
            stderr=err_pipe,
            cwd=cwd,
            env=env,
            startupinfo=si,
            shell=use_shell,
            text=process_line_by_line, # If we're processing line-by-line, then convert the shell output to a string (instead of bytes)
            close_fds=True
        )
    except FileNotFoundError as e:
        return -1, f'{e}'

    if process_line_by_line:
        return _run_with_line_processing(
            p,
            line_processor=line_processor,
            outfile=outfile,
        )

    else:
        retcode = None
        while retcode is None:
            retcode = p.poll()

        out_pipe.seek(0)
        retval = out_pipe.read()
        out_pipe.close()

        if retcode != 0:
            err_pipe.seek(0)
            retval += err_pipe.read()
        err_pipe.close()

        if not isinstance(retval, str):
            retval = retval.decode('utf-8')

        return retcode, retval


def _run_with_line_processing(p:subprocess.Popen, outfile, line_processor):
    flush_func = None
    saved_terminators = None

    if outfile is not None:
        if hasattr(outfile, 'set_terminator'):
            saved_terminators = outfile.set_terminator('')

        if hasattr(outfile, 'flush'):
            flush_func = outfile.flush

    def _write_line(line):
        if line_processor is not None:
            line = line_processor(line)
        if line and outfile is not None:
            outfile.write(line)
            if flush_func is not None:
                try:
                    flush_func()
                except:
                    pass

        return line

    retval = ''
    cancelled = False
    with SignalHandler(raise_exception_if_not_main_thread=False) as sigint:
        for out_line, err_line in _read_popen_pipes(p, sigint):
            if out_line:
                out_line = _write_line(out_line)
            if err_line:
                err_line = _write_line(err_line)

            if out_line:
                retval += out_line
            if err_line:
                retval += err_line

        if sigint.interrupted:
            cancelled = True

    if saved_terminators:
        outfile.set_terminator(saved_terminators)

    if cancelled:
        retcode = 0
    else:
        retcode = p.poll()

    return retcode, retval


def _enqueue_output(file, q):
    try:
        for line in iter(file.readline, ''):
            q.put(line)
    except KeyboardInterrupt:
        pass
    finally:
        file.close()


def _read_popen_pipes(p:subprocess.Popen, sigint:SignalHandler):
    with ThreadPoolExecutor(2) as pool:
        q_stdout, q_stderr = queue.Queue(), queue.Queue()

        pool.submit(_enqueue_output, p.stdout, q_stdout)
        pool.submit(_enqueue_output, p.stderr, q_stderr)

        while True:
            if sigint.interrupted:
                p.kill()
                break

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

            if not (out_line or err_line):
                time.sleep(0.010) # Short delay to avoid thread starvation
                continue

            yield (out_line, err_line)