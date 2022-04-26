import typer 
from .utils import (
    get_logger,
    print_info,
    print_warning,
    print_error,
    print_did_you_mean_error,
    abort,
    handle_exception,
    parse_accelerator_option,
    VariableArgumentParsingCommand,
    VariableArgumentOptionParser,
    AdditionalArgumentParsingCommand
)

# This is set at the beginning of main()
root_cli: typer.Typer = None
# MLTK build command group, this is set at the beginning of main()
build_cli: typer.Typer = None

def is_command_active() -> bool:
    """Return if a CLI command is currently active"""
    return root_cli is not None

