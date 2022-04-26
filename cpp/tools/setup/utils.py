import distutils.cmd
import logging

def get_command_logger(
    cmd: distutils.cmd.Command, 
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