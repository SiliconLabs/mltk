from typing import List, Dict
import io
import struct

try:
    import pickle
except ModuleNotFoundError:
    import pickle5 as pickle



MAX_LENGTH = ((1 << 28) - 1)


def write_data(
    pipe:io.FileIO, 
    args:List[object], 
    kwargs:Dict[str,object],
):
    tx_buf = io.BytesIO()

    data = dict( 
        args=args,
        kwargs=kwargs,
    )

    pickle.dump(data, tx_buf, protocol=pickle.HIGHEST_PROTOCOL)

    tx_bytes = tx_buf.getvalue()
    tx_len = len(tx_bytes)
    if tx_len > MAX_LENGTH:
        raise OverflowError(f'TX length ({tx_len}) > max length ({MAX_LENGTH})')
    

    tx_length_bytes = struct.pack('<L', tx_len)

    pipe.write(tx_length_bytes)
    pipe.write(tx_bytes)
    pipe.flush()


def read_data(pipe:io.FileIO) -> dict:
    rx_length_bytes = pipe.read(4)
    if not rx_length_bytes:
        return [], {}

    rx_length = struct.unpack('<L', rx_length_bytes)[0]
    if rx_length > MAX_LENGTH:
        raise OverflowError(f'RX length ({rx_length}) > max length ({MAX_LENGTH})')
    
    rx_bytes = pipe.read(rx_length)
    if not rx_bytes:
        return [], {}

    rx_buf = io.BytesIO(rx_bytes)
    rx_data = pickle.load(rx_buf)
    args = rx_data['args']
    kwargs = rx_data['kwargs']

    return args, kwargs

