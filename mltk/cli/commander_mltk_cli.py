
import typer
from mltk import cli 



@cli.root_cli.command("commander", cls=cli.VariableArgumentParsingCommand)
def silabs_commander_command(ctx: typer.Context):
    """Silab's Commander Utility

    This utility allows for accessing a Silab's embedded device via JLink.
    
    For more details issue command: mltk commander --help 
    """
    
    # Import all required packages here instead of at top
    # to help improve the CLI's responsiveness
    from mltk.utils.commander import issue_command


    logger = cli.get_logger()
    try:
        issue_command(*ctx.meta['vargs'], outfile=logger)
    except Exception as e:
        cli.handle_exception('Commander failed', e)
