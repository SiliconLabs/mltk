"""General system utilities

See the source code on Github: `mltk/utils/system.py <https://github.com/siliconlabs/mltk/blob/master/mltk/utils/system.py>`_
"""
import sys
import os
import signal
import stat
import time



def get_current_os() -> str:
    """Return the current OS handle, windows, linux or osx"""
    if sys.platform == "linux" or sys.platform == "linux2":
        return 'linux'

    elif sys.platform == "darwin":
        return 'osx'

    elif sys.platform == "win32":
        return 'windows'

    else:
        return 'unknown'


def is_windows() -> bool:
    """Return if the script is running on Windows"""
    return get_current_os() == 'windows'


def is_linux() -> bool:
    """Return if the script is running on Linux"""
    return get_current_os() == 'linux'

def is_osx() -> bool:
    """Return if the script is running on OSX"""
    return get_current_os() == 'osx'


def has_admin() -> bool:
    """Return if the script has admin permissions"""
    if os.name == 'nt':
        try:
            # only windows users with admin privileges can read the C:\windows\temp
            os.listdir(os.sep.join([os.environ.get('SystemRoot','C:\\windows'),'temp']))
        except:
            return (os.environ['USERNAME'],False)
        else:
            return (os.environ['USERNAME'],True)
    else:
        if 'SUDO_USER' in os.environ and os.geteuid() == 0: # pylint: disable=no-member
            return (os.environ['SUDO_USER'],True)
        else:
            return (os.environ['USERNAME'],False)


def get_username() -> str:
    """Return the current username (which is assumed to be the directory name of the HOME directory)"""
    home_dir = os.path.expanduser('~')
    return os.path.basename(home_dir)


def raise_signal(sig = signal.SIGINT):
    """Raise a termination signal and kill the current script"""
    os.kill(os.getpid(), sig)



def make_path_executable(path:str):
    """Set the executable permissions of the given executable path"""
    if os.name != 'nt':
        mode = os.stat(path).st_mode
        mode |= stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
        os.chmod(path, mode)


def send_signal(sig = signal.SIGINT, pid:int=None):
    """Send a signal to the current process and all children processes

    Args:
        sig: The signal to send
        pid: The process id. If None then use current process.
            If -1, only send signal to children processes of current process
            If 0, only send signal to current process
    """
    import psutil

    skip_current = False
    skip_children = False
    if pid == -1:
        skip_current = True
        pid = None
    elif pid == 0:
        skip_children = True
        pid = None

    current_process = psutil.Process(pid=pid)

    if not skip_children:
        # Attempt to kill the child processes 3 times
        for i in range(1, 4):
            children = current_process.children(recursive=True)
            if len(children) == 0:
                break
            for child in children:
                os.kill(child.pid, sig)
            if i < 3:
                time.sleep(0.100) # Wait a moment before trying again

    if not skip_current:
        os.kill(current_process.pid, sig)