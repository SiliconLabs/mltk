
import os
import sys
import re
from typing import List
from multiprocessing import current_process



# If we're in a subprocess, then disable the GPU and TF logging
# This is necessary for the parallel data generators
if current_process().name != 'MainProcess':
    # Disable the GPU and logging if this is a subprocess
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # WARNING log level
else:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' # Otherwise set the log level to debug (we use redirect_stream() to redirect the log to the cli logger)

# If it's not already specified, then  GPU uses threads dedicated to this device
# More details here:
# https://github.com/NVIDIA/DeepLearningExamples/issues/57
os.environ['TF_GPU_THREAD_MODE'] = os.environ.get('TF_GPU_THREAD_MODE', 'gpu_private') 


def main():
    """Entry point for the CLI"""

    # Determine if the tensorflow Python package should be disabled
    # This is useful for commands that don't require tensorflow,
    # as it can import startup time
    _check_disable_tensorflow()


    # We want to minimize the amount of package that 
    # get imported by subprocesses, so we manually import
    # these packages only if "main()" is actually being invoked
    import typer
    from mltk import cli 
    from mltk.utils.system import send_signal
    import signal
    import atexit
    import threading

    # Get the logger now so that we can redirect TF logs
    cli.get_logger(and_set_mltk_logger=False)

    cli.root_cli = typer.Typer(
        context_settings=dict(
            max_content_width=100
        ),
        add_completion=False
    )

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
        discover_and_import_commands(cli.root_cli)

    # This will send the CTRL+Break signal to any subprocesses after 5s of the program exiting.
    # This helps to ensure that the CLI does not hang due to processes not being cleaned up properly.
    t = threading.Timer(5, send_signal, kwargs=dict(sig=signal.SIGTERM, pid=-1))
    atexit.register(t.start)

    # Execute the command
    try:
        cli.root_cli(prog_name='mltk')
    except Exception as e:
        cli.handle_exception('Exception while executing command', e)


def discover_and_import_commands(root_cli):
    """Discover all python scripts that end with '*_mltk_cli.py'
    and import the python module"""

    from mltk import (MLTK_DIR, MLTK_ROOT_DIR)
    from mltk import cli 
    from mltk.utils.python import as_list
    from mltk.utils.path import (fullpath, get_user_setting, recursive_listdir)
    
    # This works around an issue with mltk_cli.py 
    # modules that only have one command
    # It simply adds a hidden dummy command to the root CLI
    @root_cli.command('_mltk_workaround', hidden=True)
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

            path = f'{p}/{fn}'.replace('\\', '/')
            with open(path, 'r') as f:
                file_contents = f.read()
                found_cmd_names = _find_cli_command_names(file_contents)
                if not found_cmd_names:
                    continue
                    
                # If we found the cli module corresponding to the CLI arg
                # Then immediately install the cli module and return so that we can execute it
                if cli_cmd_arg in found_cmd_names:
                    import_command(root_cli, path)
                    return

                command_paths.append(path)

    # Otherwise, if no commands matched the cli arg
    # then just import all the commands
    # (this is necessary for commands like: mltk --help)
    for path in command_paths:
        import_command(root_cli, path)


def import_command(root_cli, cli_path:str):  
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



def _version_callback(value: bool):
    """Print the version of the mltk package and exit"""
    if value:
        import typer 
        import mltk
        typer.echo(mltk.__version__)
        raise typer.Exit()


def _disable_gpu_callback(value: bool):
    """Disable usage of the GPU
    This does the same thing as defining the environment variable: CUDA_VISIBLE_DEVICES=-1
    """
    if not value:
        from mltk import cli 
        cli.print_warning('Disabling GPU')
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
