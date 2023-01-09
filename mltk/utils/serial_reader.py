from typing import List, Dict
import sys
import time
import re
import threading
import queue

import serial
import serial.tools.list_ports
from .python import as_list


class SerialReader(object):
    """Utility for reading data from a serial COM port

    Args:
        port: Name of serial COM port, if starts with "regex:" then try to find a matching port by listing all ports
        baud: Baud rate
        outfile: File-like object to write received serial data, use sys.stdout if omitted
        mode: outfile mode, if mode='r' then write as ASCII (ignore char > 127 and \\r), if mode='rb' then write as binary
        start_regex: Regex or list of Regex to use match against received serial data before writing to the captured_data buffer during read()
        stop_regex: Regex or list of Regex to use match against received serial data to stop writing to the captured_data buffer and read() returns
        fail_regex: Regex or list of Regex to use match against received serial data to abort read()

    See the source code on Github: `mltk/utils/serial_reader.py <https://github.com/siliconlabs/mltk/blob/master/mltk/utils/serial_reader.py>`_
    """
    def __init__(
        self,
        port: str,
        baud=115200,
        outfile=None,
        mode = 'r',
        start_regex: re.Pattern = None,
        stop_regex: re.Pattern = None,
        fail_regex: re.Pattern = None,
    ):
        self.port = port
        self.baud = baud
        self.mode = mode
        self.outfile = outfile or sys.stdout
        self.start_regex = start_regex
        self.stop_regex = stop_regex
        self.fail_regex = fail_regex
        self._handle : serial.Serial = None

        self._started = False
        self._stopped = False
        self._failed = False

        self._captured_data = ''
        self._error_message = ''

        self._rx_thread_active = threading.Event()
        self._rx_thread:threading.Thread = None
        self._rx_queue = queue.Queue()


    @property
    def is_open(self) -> bool:
        """Return if the serial connection is opened"""
        return self._handle is not None and self._handle.is_open

    @property
    def started(self) -> bool:
        """Return if the start_regex condition has been found"""
        return self._started

    @property
    def stopped(self) -> bool:
        """Return if the stop_regex condition has been found"""
        return self._stopped

    @property
    def failed(self) -> bool:
        """Return if the fail_regex condition has been found"""
        return self._failed

    @property
    def captured_data(self) -> str:
        """Data received by read() between the start_regex and stop_regex conditions"""
        return self._captured_data

    @property
    def error_message(self) -> str:
        """Data received after fail_regex was found"""
        return self._error_message


    @staticmethod
    def list_ports() -> List[Dict[str, str]]:
        """Retrun a list of COM ports"""
        retval = []
        for port, desc, hwid in sorted(serial.tools.list_ports.comports()):
            retval.append(dict(port=port, desc=desc, hwid=hwid))
        return retval


    @staticmethod
    def resolve_port(port: str) -> str:
        """List the COM ports and try to find the given port in the list"""
        if not port:
            raise ValueError('Null port provided')

        ports = SerialReader.list_ports()

        port_re = None
        if port.startswith('regex:'):
            port_re = re.compile(port[len('regex:'):], re.IGNORECASE)

        for port_details in ports:
            if port_re is not None:
                if port_re.match(port_details['desc']):
                    return port_details['port']
                if port_re.match(port_details['hwid']):
                    return port_details['port']
                continue

            if port_details['port'].lower() == port.lower():
                return port_details['port']

        if ports:
            available_ports = 'Available COM ports:\n' + '\n'.join([x['port'] for x in ports])
        else:
            available_ports = 'No serial COM ports available'

        raise RuntimeError(
            f'Serial COM port not found: {port}\n' \
            f'{available_ports}\n\n' \
            'Is the development board on and properly enumerated?\n' \
            'Are any other programs connected to the board\'s COM port?'
        )


    def open(self):
        """Open the a connection to the serial port"""
        port = SerialReader.resolve_port(self.port)
        if not port:
            raise Exception('Invalid serial port')

        try:
            self._handle = serial.Serial(
                port=port,
                baudrate=self.baud,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )
        except Exception as e:
            raise Exception( # pylint: disable=raise-missing-from
                f'Failed to open COM port: {port}\n' \
                'Ensure the development board is on and properly enumerated.\n' \
                'Also ensure no other serial terminals are connected to the COM port.\n' \
                f'Error details: {e}'
            )
        self.flush()

        self.start_regex = as_list(self.start_regex)
        self.stop_regex = as_list(self.stop_regex)
        self.fail_regex = as_list(self.fail_regex)
        self._captured_data = ''

        self._started = False
        self._stopped = False
        self._failed = False
        if not self.start_regex and (self.stop_regex or self.fail_regex):
            self._started = True

        self._rx_thread_active.clear()
        self._rx_thread = threading.Thread(
            name='SerialRx',
            target=self._read_loop,
            daemon=True
        )
        self._rx_thread.start()


    def close(self):
        """Close the serial COM port"""
        if  self._rx_thread is not None:
            self._rx_thread_active.set()
            self._rx_thread.join(5)
            self._rx_thread = None

        while not self._rx_queue.empty():
            self._rx_queue.get()

        if self.is_open:
            self._handle.close()
            self._handle = None


    def flush(self):
        """Flush any received data"""
        if not self.is_open:
            raise Exception('Connection not opened')

        self._handle.reset_input_buffer()
        self._handle.reset_output_buffer()
        self._captured_data = ''
        while not self._rx_queue.empty():
            self._rx_queue.get()


    def read(self, timeout:float=None) -> bool:
        """Read data for the given timeout or until stop_regex or fail_regex
        have been found in the received data.

        The captured_data property will contain the received data between the start_regex and stop_regex conditions.
        The error_message property will contain any data received after the fail_regex conditions.

        NOTE: The outfile will contain ALL received data, regardless of the start_regex/stop_regex conditions

        Args:
            timeout: Maximum time in seconds to receive serial data. If None then read until stop_regex/fail_regex found.

        Returns:
            True if the stop_regex or fail_regex were found in the received data.
            False on on timeout.
        """

        if not self.is_open:
            raise Exception('Connection not opened')

        # Wait forever if not timeout is given
        timeout = timeout or 1e9

        start_time = time.time()
        saved_terminators = None
        if hasattr(self.outfile, 'set_terminator'):
            saved_terminators = self.outfile.set_terminator('')

        try:
            while (time.time() - start_time) < timeout:
                self._buffer_data()

                if self._check_for_fail_condition():
                    return True

                if not self._wait_for_start_condition():
                    continue

                if self._check_for_stop_condition():
                    return True

        finally:
            if saved_terminators:
                self.outfile.set_terminator(saved_terminators)

        return False


    def _buffer_data(self):
        """Receive data from the COM port and write the the outfile and _captured_data buffer

        """
        if not self.is_open:
            raise Exception('Connection not opened')

        new_data = ''
        while not self._rx_queue.empty():
            data = self._rx_queue.get()
            if self.mode == 'rb':
                new_data = data
            elif self.mode == 'r':
                for d in data:
                    # Ignore non-ASCII and \r
                    if d > 127 or d == 13:
                        continue
                    new_data += chr(d)

        if new_data and self.outfile is not None:
            self.outfile.write(new_data)
            self.outfile.flush()

        if self.start_regex or self.stop_regex or self.fail_regex:
            self._captured_data += new_data


    def _read_loop(self):
        """Thread loop to read the COM port"""
        while True:
            if self._handle.in_waiting > 0:
                data = self._handle.read(self._handle.in_waiting)
                self._rx_queue.put(data)
            if self._rx_thread_active.wait(0.005):
                break


    def _wait_for_start_condition(self):
        """Determine if a start condition is found in the _captured_data buffer
        If so, reset the _captured_data buffer and set _started=True.
        This immediately returns if the start condition was previously found.
        """
        if not self.start_regex or self._started:
            return True

        found = False
        for regex in self.start_regex:
            match = regex.search(self._captured_data)
            if match is not None:
                found = True
                break

        if not found:
            return False

        self._started = True
        self._captured_data = self._captured_data[match.end(0)+1:]

        return True


    def _check_for_stop_condition(self):
        """Check if the stop_regex is found in the _captured_data buffer"""
        if not self.stop_regex:
            return False

        found = False
        for regex in self.stop_regex:
            match = regex.search(self._captured_data)
            if match is not None:
                found = True
                break

        if not found:
            return False

        self._stopped = True
        self._captured_data = self._captured_data[:match.start(0)]

        return True


    def _check_for_fail_condition(self):
        """Check if a fail_regex is found in the _captured_data buffer"""
        if not self.fail_regex:
            return False

        found = False
        for regex in self.fail_regex:
            match = regex.search(self._captured_data)
            if match is not None:
                found = True
                break

        if not found:
            return False

        self._failed = True
        start_index = match.start(0)
        self._error_message = self._captured_data[start_index:]
        return True


    def __enter__ (self):
        self.open()
        return self


    def __exit__ (self, *args, **kwargs):
        self.close()
