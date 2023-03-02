
import sys
import os
import platform
import difflib
import types
import re
import functools
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



def create_cli():
    """Create the MLTK CLI instances

    This instantiates the Typer objects:
    - cli.root_cli
    - cli.build_cli
    """
    from mltk import cli

    # Determine if the tensorflow Python package should be disabled
    # This is useful for commands that don't require tensorflow,
    # as it can import startup time
    _check_disable_tensorflow()

    cli.root_cli = typer.Typer(
        context_settings=dict(
            max_content_width=100
        ),
        add_completion=False
    )
    cli.root_cli.__call__ = functools.partial(cli.root_cli.__call__, prog_name='mltk')

    cli.build_cli = typer.Typer(
        context_settings=dict(
            max_content_width=100
        ),
        add_completion=False
    )
    cli.root_cli.add_typer(cli.build_cli, name='build', short_help='MLTK build commands')


    @cli.root_cli.callback()
    def _main(
        version: bool = typer.Option(None, '--version',
            help='Display the version of this mltk package and exit',
            show_default=False,
            callback=_version_callback,
            is_eager=True
        ),
        gpu: bool = typer.Option(True,
            help='''\b
Disable usage of the GPU.
This does the same as defining the environment variable: CUDA_VISIBLE_DEVICES=-1
Example:
mltk --no-gpu train image_example1
''',
            show_default=False,
            callback=_disable_gpu_callback
        ),
    ):
        """Silicon Labs Machine Learning Toolkit

        This is a Python package with command-line utilities and scripts to aid the development of machine learning models
        for Silicon Lab's embedded platforms.
        """

    # Only import the commands if the command was NOT:
    # mltk --version
    if not(len(sys.argv) == 2 and sys.argv[1] == '--version'):
        # Discover and import commands
        _discover_and_import_commands()



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



def _version_callback(value: bool):
    """Print the version of the mltk package and exit"""
    if value:
        typer.echo(mltk.__version__)
        raise typer.Exit()


def _disable_gpu_callback(value: bool):
    """Disable usage of the GPU
    This does the same thing as defining the environment variable: CUDA_VISIBLE_DEVICES=-1
    """
    if not value:
        print_warning('Disabling GPU')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def _check_disable_tensorflow():
    """Determine if the tensorflow Python package should be disabled
    This is useful for commands that don't require tensorflow,
    as it can import startup time
    """
    from mltk import disable_tensorflow

    disable_tf = False

    if len(sys.argv) == 1 or '--help' in sys.argv or '--version' in sys.argv:
        disable_tf = True

    else:
        cmd = None if len(sys.argv) < 2 else sys.argv[1]
        if cmd in ('commander', 'build'):
            disable_tf = True
        elif cmd in ('profile', 'summarize', 'update_params', 'view', 'classify_audio'):
            for a in sys.argv[2:]:
                if a.endswith('.tflite'):
                    disable_tf = True
                    break

    if disable_tf:
        disable_tensorflow()



def _discover_and_import_commands():
    """Discover all python scripts that end with '*_mltk_cli.py'
    and import the python module"""

    from mltk import (MLTK_DIR, MLTK_ROOT_DIR)
    from mltk import cli
    from mltk.utils.python import as_list
    from mltk.utils.path import (fullpath, get_user_setting, recursive_listdir)

    # This works around an issue with mltk_cli.py
    # modules that only have one command
    # It simply adds a hidden dummy command to the root CLI
    @cli.root_cli.command('_mltk_workaround', hidden=True)
    def _mltk_workaround():
        pass

    # Check if a command was specified on the command-line
    cli_cmd_arg = None
    if len(sys.argv) > 1:
        cli_cmd_arg = sys.argv[1]

    search_paths = as_list(get_user_setting('cli_paths'))
    env_paths = os.getenv('MLTK_CLI_PATHS', '')
    if env_paths:
        search_paths.extend(env_paths.split(os.pathsep))

    search_paths.extend([
        f'{MLTK_DIR}/cli'
    ])

    # If we're executing from the MLTK repo
    # (i.e. NOT the pip Python package)
    # then include the C++ CLI in the search path
    cpp_cli_path = f'{MLTK_ROOT_DIR}/cpp/tools/utils'
    if os.path.exists(cpp_cli_path):
        search_paths.append(cpp_cli_path)


    # Also all any apps directories that define an mltk_cli.py script
    cpp_apps_path = f'{MLTK_ROOT_DIR}/cpp/shared/apps'
    if os.path.exists(cpp_apps_path):
        app_cli_paths = []
        def _include_app_dir(p: str) -> bool:
            app_dir = os.path.dirname(p)
            if app_dir not in app_cli_paths and p.endswith('_mltk_cli.py'):
                app_cli_paths.append(app_dir)
            return False

        recursive_listdir(cpp_apps_path, regex=_include_app_dir)
        search_paths.extend(app_cli_paths)

    # Find all *_mltk_cli.py files in the search paths
    # If we find a command that matches the one provided on the command line
    # Then just import that file and return
    command_paths = []

    for p in search_paths:
        p = fullpath(p)
        if not os.path.exists(p):
            cli.print_warning(f'Invalid CLI search path: {p}')
            continue

        for fn in os.listdir(p):
            if not fn.endswith('_mltk_cli.py'):
                continue

            cli_path = f'{p}/{fn}'.replace('\\', '/')
            with open(cli_path, 'r') as f:
                file_contents = f.read()
                found_cmd_names = _find_cli_command_names(file_contents)
                if not found_cmd_names:
                    continue

                # If we found the cli module corresponding to the CLI arg
                # Then immediately install the cli module and return so that we can execute it
                if cli_cmd_arg in found_cmd_names:
                    _import_command(cli_path)
                    return

                command_paths.append(cli_path)

    # Otherwise, if no commands matched the cli arg
    # then just import all the commands
    # (this is necessary for commands like: mltk --help)
    for cli_path in command_paths:
        _import_command(cli_path)


def _import_command(cli_path:str):
    """Import the given command python module"""
    from mltk import cli
    from mltk.utils.python import import_module_at_path

    try:
        import_module_at_path(cli_path)
    except Exception as e:
        cli.handle_exception(f'Failed to import CLI python module: {cli_path}', e, no_abort=True)


def _find_cli_command_names(file_contents: str) -> List[str]:
    """Search the given file's contents and see if it specifies commands and retrieve the command names"""

    if 'command_re' not in globals():
        # Search for similar to:
        # root_cli.command('build')
        globals()['command_re'] = re.compile(r"""^.*\.command\s*\(\s*['"]([\w_]+)['"].*\).*$""")
        # Search for similar to:
        # root_cli.add_typer(kernel_tests_cli, name='kernel_tests')
        globals()['group_re'] = re.compile(r"""^.*add_typer\s*\(.*name\s*=\s*['"]([\w_]+)['"].*\)$""")

    command_re = globals()['command_re']
    group_re = globals()['group_re']

    # If see if any command groups have been defined and return those
    retval = []
    for line in file_contents.splitlines():
        match = group_re.match(line)
        if match:
            retval.append(match.group(1))

    # Otherwise, see if any command have been defined
    if not retval:
        for line in file_contents.splitlines():
            match = command_re.match(line)
            if match:
                retval.append(match.group(1))

    return retval


