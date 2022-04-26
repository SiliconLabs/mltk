import sys
import typer
from typer.core import TyperCommand
from click.parser import OptionParser
from mltk import cli 




class _VariableArgumentOptionParser(OptionParser):
    def parse_args(self, args):
        if len(args) >= 2:
            self.ctx.meta['vargs'] = args[1:]
            return super().parse_args(args[:1])
        else:
            return super().parse_args(args)

class _VariableArgumentParsingCommand(TyperCommand):
    def make_parser(self, ctx):
        """Creates the underlying option parser for this command."""
        parser = _VariableArgumentOptionParser(ctx)
        for param in self.get_params(ctx):
            param.add_to_parser(parser, ctx)
        return parser



@cli.root_cli.command("custom", cls=_VariableArgumentParsingCommand)
def custom_model_command(
    ctx: typer.Context,
    model: str = typer.Argument(..., 
        help='Name of MLTK model OR path to model specification python script',
        metavar='<model>'
    )
):
    """Custom Model Operations

    This allows for running custom-defined model commands.
    The model operations are defined in the model specification file.
    
    ----------
     Examples
    ----------
    \b
    # Run the 'visualize' custom command provided by the 
    # siamese_contrastive example model
    mltk custom siamese_contrastive visualize

    """

    # Import all required packages here instead of at top
    # to help improve the CLI's responsiveness
    from mltk.core import load_mltk_model


    # Find the MLTK Model file
    try:
        mltk_model = load_mltk_model(
            model,  
            print_not_found_err=True
        )
    except Exception as e:
        cli.handle_exception('Failed to load model', e)

    if len(mltk_model.cli.registered_commands) == 0:
        cli.abort(msg=f'Model {mltk_model.name} does not define any custom commands')

    # This works around an issue with mltk_cli.py 
    # modules that only have one command
    # It simply adds a hidden dummy command to the root CLI
    @mltk_model.cli.command('_mltk_workaround', hidden=True)
    def _mltk_workaround():
        pass 


    try:
        orig_argv = sys.argv
        sys.argv = ['mltk'] + ctx.meta['vargs']
        mltk_model.cli(prog_name=f'mltk custom {mltk_model.name}')
    except Exception as e:
        cli.handle_exception(f'Model {mltk_model.name} custom command failed', e)
    finally:
        sys.argv = orig_argv


