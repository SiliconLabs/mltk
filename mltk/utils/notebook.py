import subprocess 
import threading
import time
import signal
from IPython.core.magic import register_line_magic
from .signal_handler import SignalHandler
from .system import send_signal


@register_line_magic
def runcmd(command:str):
    """Print the output of a command to the notebook in real-time"""

    process = subprocess.Popen(
        command, 
        stdout=subprocess.PIPE, 
        shell=True, 
        stderr=subprocess.STDOUT, 
        bufsize=8, 
        close_fds=True, 
        text=False
    )

    class LinePrintingThread(threading.Thread):
        def __init__(self, process):
            threading.Thread.__init__(self, name='Line Printing Thread', daemon=True)
            self._process = process

        def run(self) -> None:
            try:
                for line in iter(process.stdout.readline, b''):
                    print(line.decode('utf-8').rstrip())
            except:
                pass


    t = LinePrintingThread(process)
    t.start()

    with SignalHandler() as sigint:
        while not sigint.interrupted:
            time.sleep(0.1)

    try:
        send_signal(signal.SIGINT, pid=process.pid)
    except Exception as e:
        print(f'Failed to input sub-process, err: {e}')
        process.kill()
    finally:
        process.stdout.close()
    process.wait()
