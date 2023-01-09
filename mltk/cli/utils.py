
import sys
import os
import platform
import difflib
import types
import re
import traceback
import logging
from typing import List
import subprocess

import typer
from typer.core import TyperCommand
from click.parser import OptionParser

import mltk
from mltk.utils.logger import add_console_logger, make_filelike, redirect_stream
from mltk.utils.logger import get_logger as get_base_logger
from mltk.utils.python import debugger_is_active
from mltk.utils.string_formatting import pretty_time_str
from mltk.utils import path
from mltk.utils.gpu import check_tensorflow_cuda_compatibility_error


# Instantiate the cli logger
if 'logger' not in globals():
    log_dir = path.create_user_dir('cli_logs')

    # If a command name is found in the cli args,
    # then use that as the log file name, otherwise just default to cli.log
    if len(sys.argv) > 1 and re.match(r'^[\w_]+$', sys.argv[1]):
        log_file = f'{log_dir}/{sys.argv[1]}.log'
    else:
        log_file = f'{log_dir}/cli.log'

    logger = get_base_logger(
        'mltk',
        level='WARN',
        log_file=log_file,
        log_file_mode='w'
    )

    def _dump_exception(cls, e):
        cls.debug(f'{e}', exc_info=e)

    _cmd_str = ' '.join(sys.argv[1:])
    logger.debug(f'Time: {pretty_time_str()}')
    logger.debug(f'Command-line: {_cmd_str}')
    logger.debug(f'Python version:  {sys.version}')
    logger.debug(f'Python path: {sys.executable}')
    logger.debug(f'Platform: {platform.platform()}')
    logger.debug(f'MLTK version: {mltk.__version__}')

    if os.path.exists(f'{mltk.MLTK_ROOT_DIR}/.git'):
        try:
            _git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=mltk.MLTK_ROOT_DIR).decode('ascii').strip()
            logger.debug(f'MLTK repo hash: {_git_hash}')
        except:
            pass
    logger.dump_exception = types.MethodType(_dump_exception, logger)


def get_logger(verbose=False, and_set_mltk_logger=True) -> logging.Logger:
    """Get the command-line interface logger"""

    # Redirect before doing anything else
    # this way we can capture all logs from TF
    if '_is_redirecting_stderr' not in globals():
        # Redirect stderr to the logger
        # This way we can capture internal Tensorflow logs
        globals()['_is_redirecting_stderr'] = True
        try:
            redirect_stream(logger, stream='stderr')
        except Exception as e:
            logger.warning(f'Failed to redirect stderr to logger, err: {e}')

    add_console_logger(logger)
    make_filelike(logger)

    if verbose:
        logger.verbose = True

    if and_set_mltk_logger:
        # NOTE: We import here to avoid circular imports
        from ..core.utils import set_mltk_logger
        set_mltk_logger(logger)

    return logger


def print_info(msg:str):
    """Print a informational msg to the console"""
    typer.echo(msg)


def print_warning(msg:str):
    """Print a warning msg to the console"""
    typer.echo(typer.style(msg, fg=typer.colors.YELLOW))


def print_error(msg:str):
    """Print an error msg to the console"""
    typer.echo(typer.style(msg, fg=typer.colors.RED))


def print_did_you_mean_error(
    msg:str,
    not_found_arg:str,
    possible_args:List[str],
    n=15,
    cutoff=0.2,
    and_exit=False,
    prologue:str=None
):
    """Print an error message with a list of possible alternative suggestions"""
    possible_matches = difflib.get_close_matches(not_found_arg, possible_args, n=n, cutoff=cutoff)
    msg = f'{msg}: {not_found_arg}'
    if possible_matches:
        msg += '\n\nDid you mean?\n'
        msg += '\n'.join(possible_matches)
        msg += '\n'
    elif possible_args:
        msg += '\n\nPossible options:\n'
        msg += ', '.join(possible_args[:min(n, len(possible_args))])
        msg += '\n'
    else:
        msg += '\nNo options available\n'

    if prologue:
        msg += prologue

    print_error(msg)
    if and_exit:
        abort()


def abort(code=-1, msg=None):
    if msg:
        print_error(msg)
    sys.exit(code)


def handle_exception(msg: str, e: Exception, print_stderr=False, no_abort=False):
    """Handle an exception trigger while a cli command was executing"""

    # If the exception args has just 1 string value, then use that
    if hasattr(e, 'args') and isinstance(e.args, tuple) and len(e.args) == 1 and isinstance(e.args[0], str):
        err_msg = e.args[0]
    else:
        # Otherwise, try to format the string
        err_msg = f'{e}'

    gpu_err = check_tensorflow_cuda_compatibility_error(log_file)
    if gpu_err:
        err_msg += f'\n\n{gpu_err}\n\n'

    logger.debug(msg, exc_info=e)
    logger.error(f'{msg}, err: {err_msg}')

    if debugger_is_active():
        traceback.print_tb(e.__traceback__)

    if print_stderr:
        sys.stderr.write(f'{msg}, err: {err_msg}, for more details see: {log_file}')
    else:
        print_error(f'For more details see: {log_file}')

    if debugger_is_active():
        breakpoint()

    if not no_abort:
        try:
            file_handler = logger.file_handler
            if file_handler is not None:
                file_handler.close()
        except:
            pass
        abort()


def parse_accelerator_option(accelerator: str) -> str:
    """Normalize the accelerator argument, check if the given accelerator is supported.
    Print a meaningful error msg if not"""
    #pylint: disable=import-outside-toplevel
    from mltk.core.tflite_micro import TfliteMicro

    if not accelerator:
        return None

    norm_accelerator = TfliteMicro.normalize_accelerator_name(accelerator)
    if norm_accelerator is None:
        print_did_you_mean_error('Unknown accelerator', accelerator, TfliteMicro.get_supported_accelerators(), cutoff=.9999, and_exit=True)

    return norm_accelerator



class AdditionalArgumentOptionParser(OptionParser):
    variable_re = re.compile(r'^([A-Za-z0-9_\.]+)=(.+$)')

    def parse_args(self, args):
        parsed_args = []
        self.ctx.meta['additional_variables'] = {}
        self.ctx.meta['additional_args'] = []

        for i, arg in enumerate(args):
            match = self.variable_re.match(arg)
            if match:
                key = match.group(1)
                value = match.group(2)
                self.ctx.meta['additional_variables'][key] = value
                continue

            if arg == '--':
                self.ctx.meta['additional_args'] = [x for x in args[i+1:]]
                break
            parsed_args.append(arg)

        return super(AdditionalArgumentOptionParser, self).parse_args(parsed_args)


class AdditionalArgumentParsingCommand(TyperCommand):
    def make_parser(self, ctx):
        """Creates the underlying option parser for this command."""
        parser = AdditionalArgumentOptionParser(ctx)
        for param in self.get_params(ctx):
            param.add_to_parser(parser, ctx)
        return parser


class VariableArgumentOptionParser(OptionParser):
    def parse_args(self, args):
        self.ctx.meta['vargs'] = args
        return super(VariableArgumentOptionParser, self).parse_args([])


class VariableArgumentParsingCommand(TyperCommand):
    def make_parser(self, ctx):
        """Creates the underlying option parser for this command."""
        parser = VariableArgumentOptionParser(ctx)
        for param in self.get_params(ctx):
            param.add_to_parser(parser, ctx)
        return parser
