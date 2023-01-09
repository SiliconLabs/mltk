
"""
This script requires the corresponding firmware to be programmed to the dev board.
To program the firmware, add the following to the file:
# <mltk repo root>/user_options.cmake:
#
# mltk_set(MLTK_TARGET mltk_uart_stream_data_test)
#
# The invoke the CMake target:
# mltk_uart_stream_data_test_download
#
# This will build and download the firmware to the dev board.
# Then, run then run this script: python data_test.py

"""




import binascii
import logging
import time
from collections import deque

from mltk.utils.uart_stream import UartStream



logging.basicConfig(level='INFO',  handlers=[logging.StreamHandler()])
logger = logging.getLogger()
logger.info('Starting data test')


with UartStream() as stream:
    logger.info('UART stream opened')

    tx_counter = 0
    rx_counter = 0
    tx_cmd_counter = 0
    rx_cmd_counter = 0
    tx_cmd_data_counter = 0
    rx_cmd_data_counter = 0
    tx_timestamp = 0
    rx_timestamp = 0
    tx_rates = deque(maxlen=10)
    rx_rates = deque(maxlen=10)

    loop_counter = 0
    while True:
        loop_counter += 1
        if not stream.synchronize():
            tx_counter = 0
            rx_counter = 0
            tx_cmd_counter = 0
            rx_cmd_counter = 0
            tx_cmd_data_counter = 0
            rx_cmd_data_counter = 0
            time.sleep(0.01)
            continue

        if stream.rx_bytes_available > 128:
            data = stream.read_all()
            now = time.perf_counter()
            rate = len(data) / ((now - rx_timestamp) * 1000)
            rx_rates.append(rate)
            rx_avg = sum(rx_rates) / len(rx_rates)
            rx_timestamp = now
            logger.info(f'Read data: len={len(data)} ({rx_avg}kbps)')

            for c in data:
                if c != rx_counter:
                    logger.info(binascii.hexlify(data))
                    assert False, 'data error'
                rx_counter = (rx_counter + 1) % 256

        if stream.tx_bytes_available > 128:        
            data = bytearray()
            for _ in range(min(stream.tx_bytes_available, 2048)):
                data.append(tx_counter)
                tx_counter = (tx_counter + 1) % 256

            stream.write_all(data)
            now = time.perf_counter()
            rate = len(data) / (( now - tx_timestamp) * 1000)
            tx_rates.append(rate)
            tx_avg = sum(tx_rates) / len(tx_rates)
            tx_timestamp = now
            logger.info(f'Wrote data: len={len(data)} ({tx_avg}kbps)')


        if loop_counter % 10 == 0:
            payload = bytearray()
            for i in range(6):
                payload.append(tx_cmd_data_counter)
                tx_cmd_data_counter = (tx_cmd_data_counter + 1) % 256
            stream.write_command(tx_cmd_counter, payload)
            logger.info(f'Wrote cmd: {tx_cmd_counter}')
            tx_cmd_counter = (tx_cmd_counter + 1) % 256

        rx_cmd, rx_cmd_data = stream.read_command()
        if rx_cmd is not None:
            assert rx_cmd == rx_cmd_counter
            logger.info(f'Read cmd: {rx_cmd}')
            rx_cmd_counter = (rx_cmd_counter + 1) % 256
            for i in range(6):
                assert rx_cmd_data[i] == rx_cmd_data_counter
                rx_cmd_data_counter = (rx_cmd_data_counter + 1) % 256

        stream.wait(timeout=0.100)