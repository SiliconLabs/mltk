
import typer
from mltk import cli

@cli.root_cli.command('tensorboard')
def view_model_command(
    model: str = typer.Argument(..., 
        help='''\b
One of the following:
- Name of MLTK model
- Path to MLTK model's python script''',
        metavar='<model>'
    ),
    host:str = typer.Option('localhost', '-h', '--host',
        help='Local interface to start HTTP server',
        metavar='<host>'
    ),
    port:int = typer.Option(6002, '-p', '--port',
        help='Listen port of HTTP server used to view graph',
        metavar='<port>'
    ),
    launch: bool = typer.Option(True, 
        help='Automatically open a webbrowser to the Tensorboard GUI'
    ),
    verbose: bool = typer.Option(False, '--verbose', '-v', 
        help='Enable verbose console logs'
    )
):
    """Start Tensorboard for the given model

    \b
    In machine learning, to improve something you often need to be able to measure it. 
    TensorBoard is a tool for providing the measurements and visualizations needed during the machine learning workflow. 
    It enables tracking experiment metrics like loss and accuracy, visualizing the model graph,
    projecting embeddings to a lower dimensional space, and much more.
    \b
    For more details, see:
    https://www.tensorflow.org/tensorboard/get_started
    \b
    NOTE: The model must be trained (or actively being trained) before using this command.
    Additionally, the trained model must have the 'tensorboard' property configured.
    e.g.:
    my_model.tensorboard = dict(
        histogram_freq=1,       # frequency (in epochs) at which to compute activation and weight histograms 
                                # for the layers of the model. If set to 0, histograms won't be computed. 
                                # Validation data (or split) must be specified for histogram visualizations.
        write_graph=True,       # whether to visualize the graph in TensorBoard. The log file can become quite large when write_graph is set to True.
        write_images=False,     # whether to write model weights to visualize as image in TensorBoard.
        update_freq="epoch",    # 'batch' or 'epoch' or integer. When using 'batch', writes the losses and metrics 
                                # to TensorBoard after each batch. The same applies for 'epoch'. 
                                # If using an integer, let's say 1000, the callback will write the metrics and losses 
                                # to TensorBoard every 1000 batches. Note that writing too frequently to 
                                # TensorBoard can slow down your training.
        profile_batch=2,        # Profile the batch(es) to sample compute characteristics. 
                                # profile_batch must be a non-negative integer or a tuple of integers. 
                                # A pair of positive integers signify a range of batches to profile. 
                                # By default, it will profile the second batch. Set profile_batch=0 to disable profiling.
    ) 
    
    \b
    ----------
     Examples
    ----------
    \b
    # Start tensorboard for the previously trained keyword_spotting_on_ff_v2 model
    # This will open a webpage to the local Tensorboard GUI
    mltk tensorboard keyword_spotting_on_ff_v2
    \b
    # Start training the audio_example1 model
    mltk train audio_example1
    # In a separate terminal, start the tensorboard GUI
    mltk tensorboard audio_example1

    """
    import os
    import webbrowser
    import time
    from mltk.core import load_mltk_model
    from mltk.utils.logger import get_logger

    logger = cli.get_logger(verbose=verbose)
    get_logger('tensorboard', console=False, base_level='WARNING', parent=logger)

    try:
        mltk_model = load_mltk_model(model, print_not_found_err=True)
    except Exception as e:
        cli.handle_exception('Failed to load MLTK model', e)

    tb_log_dir = mltk_model.log_dir + '/train/tensorboard'

    if not os.path.exists(tb_log_dir):
        cli.abort(
            msg=f'The Tensorboard log directory for the model: {model}\ndoes not exist at: {tb_log_dir}\n'
'''
Ensure the model defines the "tensorboard" property, e.g.:
my_model.tensorboard = dict(
            histogram_freq=1,       # frequency (in epochs) at which to compute activation and weight histograms 
                                    # for the layers of the model. If set to 0, histograms won't be computed. 
                                    # Validation data (or split) must be specified for histogram visualizations.
            write_graph=True,       # whether to visualize the graph in TensorBoard. The log file can become quite large when write_graph is set to True.
            write_images=False,     # whether to write model weights to visualize as image in TensorBoard.
            update_freq="epoch",    # 'batch' or 'epoch' or integer. When using 'batch', writes the losses and metrics 
                                    # to TensorBoard after each batch. The same applies for 'epoch'. 
                                    # If using an integer, let's say 1000, the callback will write the metrics and losses 
                                    # to TensorBoard every 1000 batches. Note that writing too frequently to 
                                    # TensorBoard can slow down your training.
            profile_batch=2,        # Profile the batch(es) to sample compute characteristics. 
                                    # profile_batch must be a non-negative integer or a tuple of integers. 
                                    # A pair of positive integers signify a range of batches to profile. 
                                    # By default, it will profile the second batch. Set profile_batch=0 to disable profiling.
) 

and that the model has been previously trained or is actively being trained,\ne.g.: first run the command:
'''
f'mltk train {model}\n'
        )
    logger.info(f'Tensorboard model logdir: {tb_log_dir}')
    
    from tensorboard import default
    from tensorboard import program
    try:
        import tensorboard_plugin_profile
    except:
        raise RuntimeError('Failed import tensorboard_plugin_profile Python package, try running: pip install tensorboard_plugin_profile OR pip install silabs-mltk[full]')

    try:
        tb = program.TensorBoard(
            plugins=default.get_plugins()
        )
        tb.configure(argv=[None, '--logdir', tb_log_dir, '--host', host, '--port', str(port)])
        if launch:
            url = tb.launch()
            logger.info(f'Opening {url} (Press Ctrl+C to exit)')
            webbrowser.open_new_tab(url)

            while True:
                try:
                    time.sleep(1)
                except KeyboardInterrupt:
                    break
        else:
            tb.main()

    except Exception as e:
        cli.handle_exception('Failed to run Tensorboard', e)



