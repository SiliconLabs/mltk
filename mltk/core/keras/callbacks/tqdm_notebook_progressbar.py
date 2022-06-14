import logging
from mltk.utils.python import install_pip_package, notebook_is_active

def import_tqdm_progressbar_callback(logger: logging.Logger = None):
    """Attempt to import the TQDM Jupyter Notebook Progressbar Callback
    
    https://www.tensorflow.org/addons/tutorials/tqdm_progress_bar

    NOTE: This will attempt to automatically install the Python packages

    Returns:
        TQDMProgressBar class if successfully imported, None else 
    """
    if not notebook_is_active():
        return None 

    try:
        install_pip_package('tensorflow-addons', 'tensorflow_addons', upgrade=True, logger=logger)
        install_pip_package('ipywidgets', logger=logger)
        install_pip_package('tqdm>=4.36.1')

        from tqdm import tqdm_notebook
        import tensorflow_addons as tfa
        cb = tfa.callbacks.TQDMProgressBar

        def _get_tqdm(self):
            return tqdm_notebook
        def _set_tqdm(self, v):
            pass 
        cb.tqdm = property(_get_tqdm, _set_tqdm)
        return cb
    except Exception as e:
        #logger.error('Failed to import tqdm', exc_info=e)
        return None
