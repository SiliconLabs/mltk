"""Data logging utilities

See the source code on Github: `mltk/utils/logger.py <https://github.com/siliconlabs/mltk/blob/master/mltk/utils/logger.py>`_
"""
import logging
import os
import re
import io
import sys
import types
import time
import atexit
import threading
from typing import Callable, Union, TextIO


def get_logger(
    name='mltk',
    level='INFO',
    console=False,
    log_file=None,
    log_file_mode='w',
    parent:logging.Logger=None,
    base_level='DEBUG',
    file_level='DEBUG'
):
    """Get or create a logger, optionally adding a console and/or file handler"""
    logger = logging.getLogger(name)
    if len(logger.handlers) == 0:
        if parent is None:
            logger.propagate = False
        else:
            logger.parent = parent
            logger.propagate = True

        logger.setLevel(base_level)

        if console:
            add_console_logger(logger, level=level)

        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            fh = logging.FileHandler(log_file, mode=log_file_mode)
            fh.setLevel(file_level)
            logger.addHandler(fh)

    if not hasattr(logger, 'close'):
        def _close(cls):
            for handler in cls.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
        logger.close = types.MethodType(_close, logger)

    return logger



def add_console_logger(logger: logging.Logger, level='INFO'):
    """Add a console logger to the given logger"""
    for handler in logger.handlers:
        if isinstance(handler, _ConsoleStreamLogger):
            return

    ch = _ConsoleStreamLogger(sys.stdout)
    ch.setLevel(get_level(level))
    logger.addHandler(ch)


def make_filelike(logger: logging.Logger, level=logging.INFO):
    """Make the given logger 'file-like'"""
    # pylint: disable=protected-access
    if logger is None:
        return

    # Convert the level to an int
    level = logging._nameToLevel[get_level(level)]
    logger._buffer = ''

    def _isatty(cls):
        return False

    def _write(cls, data):
        cls._buffer +=  data

    def _flush(cls):
        if cls._buffer:
            cls.log(level, cls._buffer)
            cls._buffer = ''
            for h in logger.handlers:
                h.flush()

    def _get_terminator(cls):
        retval = []
        for h in logger.handlers:
            retval.append(h.terminator)
        return retval

    def _set_terminator(cls, terminator):
        previous_terminators = _get_terminator(cls)
        if not isinstance(terminator, (list,tuple)):
            terminator = [terminator] * len(previous_terminators)

        for i, h in enumerate(logger.handlers):
            h.terminator = terminator[i]

        return previous_terminators


    if not hasattr(logger, 'write'):
        logger.write = types.MethodType(_write, logger)
    if not hasattr(logger, 'flush'):
        logger.flush = types.MethodType(_flush, logger)
    if not hasattr(logger, 'isatty'):
        logger.isatty = types.MethodType(_isatty, logger)
    if not hasattr(logger, 'set_terminator'):
        logger.set_terminator = types.MethodType(_set_terminator, logger)
    if not hasattr(logger, 'get_terminator'):
        logger.get_terminator = types.MethodType(_get_terminator, logger)


def redirect_stream(
    logger:logging.Logger,
    stream:Union[TextIO,str]='stderr',
    close_atexit=True
) -> Callable:
    """Redirect std logs to the given logger

    NOTE: This redirects ALL logs from the stream
    """

    saved_sys_std_stream_name = None
    if isinstance(stream, str):
        saved_sys_std_stream_name = stream
        stream = getattr(sys, saved_sys_std_stream_name)

    stream_fd = stream.fileno()
    saved_stream_fd = os.dup(stream_fd)
    read_stream, write_stream = os.pipe()
    os.dup2(write_stream, stream_fd)
    os.close(write_stream)

    if saved_sys_std_stream_name is not None:
        if saved_sys_std_stream_name == 'stderr':
            setattr(sys, saved_sys_std_stream_name, sys.stdout)
        else:
            setattr(sys, saved_sys_std_stream_name, io.TextIOWrapper(os.fdopen(stream_fd, 'wb'), encoding='utf-8' ))

    # Map tensorflow logs to the Python logger's corresponding level
    tf_err_re = re.compile('.*:\s([DIWE])\s(.*)')
    def _process_line(line:str):
        match = tf_err_re.match(line)
        if match:
            level_str = match.group(1)
            # We only want to print TF errors at the error level
            # Everything else to too verbose so we just map them to debug
            if level_str == 'E':
                level = logging.ERROR
            else:
                level = logging.DEBUG
            logger.log(level, match.group(2))
        else:
            logger.debug(line)

    def _drain_pipe():
        line = ''
        while True:
            data = os.read(read_stream, 256).decode('utf-8')
            if not data:
                break
            for c in data:
                line += c
                if c == '\n':
                    _process_line(line.strip())
                    line = ''

    pipe_redirect_thread = threading.Thread(
        target=_drain_pipe,
        name='stream_redirect',
        daemon=True
    )

    def _close_pipe():
        try:
            os.close(stream_fd)
        except:
            pass
        pipe_redirect_thread.join(timeout=1)

        try:
            os.close(read_stream)
        except:
            pass
        try:
            os.dup2(saved_stream_fd, stream_fd)
        except:
            pass

        try:
            os.close(saved_stream_fd)
        except:
            pass

    if close_atexit:
        atexit.register(_close_pipe)

    pipe_redirect_thread.start()

    return _close_pipe



def timing_decorator(f, level='INFO'):
    """Print the run-time of the decorated function to the logger

    If a logger is found in the args then that is used,
    else if a logger is found in the 'self' argument, then that is used
    """
    def wrap(*args, **kwargs):
        logger = None
        if logger is None:
            for a in args:
                if isinstance(a, logging.Logger):
                    logger = a
                    break
        if logger is None:
            for a in kwargs.values():
                if isinstance(a, logging.Logger):
                    logger = a
                    break
        if logger is None and len(args) > 0:
            self = args[0]
            for key in dir(self):
                value = getattr(self, key)
                if isinstance(value, logging.Logger):
                    logger = value
                    break

        if logger is None:
            logger = get_logger()

        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        elapsed = te - ts
        logger.log(get_level(level), f'{f.__name__} took: {elapsed}s')
        return result
    return wrap


def set_console_level(logger:logging.Logger, level:str) -> str:
    """Set the logger's console level and return the previous level"""
    if level is None:
        return None

    prev_console_level = None
    if hasattr(logger, 'console_level'):
        prev_console_level = logger.console_level
        logger.console_level = level
    return prev_console_level


def get_level(level:Union[str,int]) -> str:
    """Return the logging level as a string"""
    if isinstance(level, str):
        return level.upper()
    return logging.getLevelName(level)


class ConsoleLoggerLevelContext:
    def __init__(self, logger:logging.Logger, level:str):
        self.logger = logger
        self.level = level

    def __enter__(self):
        self.saved_console_level = self.logger.console_level
        self.logger.console_level = self.level

    def __exit__(self ,type, value, traceback):
        self.logger.console_level = self.saved_console_level



class DummyLogger():
    def __init__(self):
        self.handlers = []

    def debug(self, *args, **kwargs):
        pass
    def info(self, *args, **kwargs):
        pass
    def warning(self, *args, **kwargs):
        pass
    def error(self, *args, **kwargs):
        pass
    def exception(self, *args, **kwargs):
        pass
    def write(self, *args, **kwargs):
        pass
    def flush(self, *args, **kwargs):
        pass


# This is needed to distinguish between a FileHandler and console StreamHandler
class _ConsoleStreamLogger(logging.StreamHandler):
    pass


def _get_verbose(self):
    for h in self.handlers:
        if isinstance(h, _ConsoleStreamLogger):
            return h.level == logging.DEBUG
    return False

def _set_verbose(self, value : bool):
    level = 'DEBUG' if value else 'INFO'
    for h in self.handlers:
        if isinstance(h, _ConsoleStreamLogger):
            h.setLevel(level)

def _set_console_log_level(self, level:str):
    for h in self.handlers:
        if isinstance(h, _ConsoleStreamLogger):
            h.setLevel(get_level(level))

def _get_console_log_level(self):
    for h in self.handlers:
        if isinstance(h, _ConsoleStreamLogger):
            return h.level
    return None


def _get_file_handler(self):
    for h in self.handlers:
        if isinstance(h, logging.FileHandler):
            return h
    return None

logging.Logger.verbose = property(_get_verbose, _set_verbose, doc='Enable/disable verbose logging to the console')
logging.Logger.console_level = property(_get_console_log_level, _set_console_log_level, doc='Get/set the logger console logging level')
logging.Logger.file_handler = property(_get_file_handler, doc='Get the logger\'s file handler')

