import os
import signal


class SignalHandler(object):
    """Class to detect OS signals
    
    e.g. detect when CTRL+C is pressed and issue a callback
    """
    def __init__(self, sig=signal.SIGINT, callback=None, resignal_on_exit=False):
        self.sig = sig
        self.interrupted = False
        self.released = False
        self.original_handler = None
        self.resignal_on_exit = resignal_on_exit
        self.callback = callback 
    

    def __enter__(self):
        self.interrupted = False
        self.released = False

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

        signal.signal(self.sig, _handler)

        return self


    def __exit__(self, t, value, tb):
        self.release()
        if self.interrupted and self.resignal_on_exit:
            os.kill(os.getpid(), self.sig)


    def release(self):
        if self.released:
            return False

        signal.signal(self.sig, self.original_handler)

        self.released = True

        return True