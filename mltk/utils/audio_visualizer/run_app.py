import logging 

from .install_wxpython import install_wxpython


def run(model:str=None, logger:logging.Logger=None):
    """Start the AudioFeatureGenerator Visualization GUI"""

    install_wxpython(logger=logger)

    # Import these here in-case wxPython needs to be installed
    from .audio_visualizer import AudioVisualizer
    from .app import VisualizerApp

    visualizer = AudioVisualizer.instance()
    if logger:
        visualizer.logger = logger
    app = VisualizerApp()
    if model is not None:
        visualizer.load_model(model)

    if logger:
        logger.info('Starting GUI ...')
    app.MainLoop()