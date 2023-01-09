import os
import signal
import threading


class SignalHandler(object):
    """Class to detect OS signals

    e.g. detect when CTRL+C is pressed and issue a callback

    See the source code on Github: `mltk/utils/signal_handler.py <https://github.com/siliconlabs/mltk/blob/master/mltk/utils/signal_handler.py>`_
    """
    def __init__(
        self,
        sig=signal.SIGINT,
        callback=None,
        resignal_on_exit=False,
        raise_exception_if_not_main_thread=True
    ):
        self.sig = sig
        self.interrupted = False
        self.released = False
        self.original_handler = None
        self.resignal_on_exit = resignal_on_exit
        self.raise_exception_if_not_main_thread = raise_exception_if_not_main_thread
        self.callback = callback


    def __enter__(self):
        self.interrupted = False
        self.released = False

        is_main_thread = threading.current_thread() is threading.main_thread()
        if self.raise_exception_if_not_main_thread and not is_main_thread:
            raise RuntimeError('SignalHandler may only be used in the "main" thread')

        if is_main_thread:
            self.original_handler = signal.getsignal(self.sig)

        def _handler(signum, frame):
            forward_signal = False
            if not self.interrupted:
                if self.callback:
                    try:
                        forward_signal = self.callback() == 'forward-signal'
                    except:
                        pass
                self.interrupted = True
            self.release()

            if forward_signal:
                self.original_handler()

        if is_main_thread:
            signal.signal(self.sig, _handler)

        return self


    def __exit__(self, t, value, tb):
        self.release()


    def release(self):
        if self.released:
            return False

        if self.original_handler is not None:
            signal.signal(self.sig, self.original_handler)

        self.released = True
        if self.interrupted and self.resignal_on_exit:
            os.kill(os.getpid(), self.sig)

        return True