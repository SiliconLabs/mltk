import logging
from setuptools import Command
try:
    # Older python versions require this
    import distutils
    have_disutils = True
except:
    have_disutils = False



def get_command_logger(
    cmd: Command, 
    name='setup', 
    level=logging.DEBUG
):
    """Add a Python logger to the setup command class"""
    class _Handler(logging.StreamHandler):
        def __init__(self, announce):
            super().__init__()
            self.announce = announce
        def emit(self, record:logging.LogRecord):
            msg = self.format(record).rstrip()

            if have_disutils:
                if record.levelno == logging.DEBUG:
                    level = distutils.log.DEBUG
                elif record.levelno == logging.INFO:
                    level = distutils.log.INFO
                elif record.levelno == logging.WARNING:
                    level = distutils.log.WARN
                elif record.levelno == logging.ERROR:
                    level = distutils.log.ERROR
            self.announce(msg, level=level)

    logger = logging.Logger(name, level=logging.DEBUG)
    logger.addHandler(_Handler(cmd.announce))
    return logger