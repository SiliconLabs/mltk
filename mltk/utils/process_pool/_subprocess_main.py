import sys
import os
import traceback

stdin = open(sys.stdin.fileno(), 'rb')
stdout = open(sys.stdout.fileno(), 'wb')
sys.stdout = sys.stderr

from mltk.utils.python import import_module_at_path
from mltk.utils.process_pool._utils import (read_data, write_data)


MODULE_PATH = os.environ['MODULE_PATH']
FUNCTION_NAME = os.environ['FUNCTION_NAME']
PROCESS_POOL_NAME = os.environ['PROCESS_POOL_NAME']


try:
    module_instance = import_module_at_path(MODULE_PATH)
except KeyboardInterrupt:
    sys.exit(0)
except Exception as e:
    print(f'{PROCESS_POOL_NAME}: Failed to import {MODULE_PATH}, err:\n{e}')
    sys.exit(-1)

try:
    function_instance = getattr(module_instance, FUNCTION_NAME)
except KeyboardInterrupt:
    sys.exit(0)
except Exception as e:
    print(f'{PROCESS_POOL_NAME}: Failed to retrieve {FUNCTION_NAME} for {MODULE_PATH}, err:\n{e}')
    sys.exit(-1)


def main():
    while True:
        args, kwargs = read_data(stdin)
        if not args and not kwargs:
            return
        
        tx_data = function_instance(*args, **kwargs)

        if isinstance(tx_data, (tuple,list)):
            args = tx_data
        else:
            args = (tx_data,)

        write_data(stdout, args, {})



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        traceback.print_exc()
        print(f'{PROCESS_POOL_NAME}: Exception in main loop, err:\n{e}', file=sys.stderr, flush=True)
        sys.stderr.flush()
        sys.exit(-1)

    sys.exit(0)
