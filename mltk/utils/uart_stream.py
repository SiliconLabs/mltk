from __future__ import annotations
from typing import List, Dict, NamedTuple
import threading
import re
import time
import struct
import logging
import binascii
from dataclasses import dataclass

import serial
import serial.tools.list_ports


logger = logging.getLogger(__file__)


class UartStream:
    """Allows for streaming binary data between a Python script and embedded device via UART

    Features:

    - Asynchronous reception of binary data
    - Data flow control
    - C++ library (see <mltk repo>/cpp/shared/uart_stream)
    - Send/receive "commands"


    See the source code on Github: `mltk/utils/uart_stream.py <https://github.com/siliconlabs/mltk/blob/master/mltk/utils/uart_stream.py>`_

    See the corresponding example C++ app on Github: `cpp/shared/uart_stream/examples/data_test <https://github.com/siliconlabs/mltk/blob/master/cpp/shared/uart_stream/examples/data_test>`_


    Args:
        port: Name of serial COM port, if starts with "regex:" then try to find a matching port by listing all ports
        baud: Baud rate
        rx_buffer_length: Size of the RX buffer in bytes
        open_synchronize_timeout: Number of seconds to wait for the link to synchronize with the device

    """

    def __init__(
        self,
        port:str = 'regex:JLink CDC UART Port',
        baud:int = 115200,
        rx_buffer_length:int = 4096,
        open_synchronize_timeout:bool = 60.0,
    ):
        self._port = port
        self._baud = baud
        self._handle : serial.Serial = None
        self._rx_thread_active = threading.Event()
        self._rx_thread:threading.Thread = None
        self._rx_buffer = bytearray()
        self._rx_buffer_length = rx_buffer_length
        self._rx_previous_data_packet_id = 0
        self._rx_cmd_code:int = None
        self._rx_cmd_payload:bytes = None

        self._lock = threading.RLock()
        self._condition = threading.Condition(lock=self._lock)

        self._tx_bytes_available = -1
        self._tx_next_packet_id = 1
        self._tx_active_packet_id = 0

        self._sync_requested = False
        self._open_synchronize_timeout = open_synchronize_timeout

    @property
    def baud(self) -> int:
        """Serial BAUD rate"""
        return self._baud

    @property
    def port(self) -> str:
        """Serial port"""
        return self._port

    @property
    def is_open(self) -> bool:
        """Return if the serial connection is opened"""
        with self._lock:
            return not self._rx_thread_active.is_set() and self._handle is not None and self._handle.is_open

    @property
    def rx_bytes_available(self) -> int:
        """The number of received bytes in the RX buffer, ready to read by the application"""
        with self._lock:
            if not self._sync_requested and self._tx_bytes_available >= 0:
                return len(self._rx_buffer)
            else:
                return -1

    @property
    def tx_bytes_available(self) -> int:
        """The maximum of bytes that may be immediately written by the application"""
        with self._lock:
            if self._sync_requested:
                return -1
            if self._tx_active_packet_id > 0:
                return 0

            return self._tx_bytes_available

    @property
    def is_synchronized(self) -> bool:
        """Return if the data link is synchronized"""
        with self._lock:
            return not self._sync_requested and self._tx_bytes_available >= 0


    @staticmethod
    def list_ports() -> List[Dict[str, str]]:
        """Return a list of COM ports"""
        retval = []
        for port, desc, hwid in sorted(serial.tools.list_ports.comports()):
            retval.append(dict(port=port, desc=desc, hwid=hwid))
        return retval


    @staticmethod
    def resolve_port(port: str) -> str:
        """List the COM ports and try to find the given port in the list"""
        if not port:
            raise ValueError('Null port provided')

        ports = UartStream.list_ports()

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


    def open(self, timeout:float=None):
        """Open the a connection to the serial port"""

        with self._lock:
            if self.is_open:
                raise RuntimeError('Serial connection already opened')

            port = UartStream.resolve_port(self._port)
            if not port:
                raise Exception('Invalid serial port')

            logger.debug(f'Opening {port}')

            try:
                self._handle = serial.Serial(
                    port=port,
                    baudrate=self._baud,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    bytesize=serial.EIGHTBITS,
                    write_timeout=30.0,
                )
            except Exception as e:
                raise Exception( # pylint: disable=raise-missing-from
                    f'Failed to open COM port: {port}\n' \
                    'Ensure the development board is on and properly enumerated.\n' \
                    'Also ensure no other serial terminals are connected to the COM port.\n' \
                    f'Error details: {e}'
                )
            self._handle.reset_input_buffer()
            self._handle.reset_output_buffer()

            self._tx_active_packet_id = 0
            self._rx_buffer = bytearray()
            self._rx_thread_active.clear()
            self._rx_thread = threading.Thread(
                name='UartStreamRx',
                target=self._rx_loop,
                daemon=True
            )
            self._rx_thread.start()
            logger.debug('Opened')

        timeout = timeout or self._open_synchronize_timeout
        if timeout:
            self.synchronize_blocking(timeout=timeout)


    def close(self):
        """Close the serial COM port"""
        if  self._rx_thread is not None:
            self._rx_thread_active.set()
            self._rx_thread.join(5)
            self._rx_thread = None
            self._rx_buffer = bytearray()
            self._tx_bytes_available = -1

        with self._lock:
            if self.is_open:
                self._handle.close()
                self._handle = None


    def synchronize(self, timeout:float=None, ack_read_data=True) -> bool:
        """Synchronize UART link with other size

        Before data transfer may behind, both sides need to "synchronize".
        This should be periodically called until it returns true indicating
        that the link is sync'd.

        NOTE: This is non-blocking
        """
        if not self.is_open:
            raise RuntimeError('Connection not opened')

        with self._lock:
            sync_requested = self._sync_requested
            self._sync_requested = False

            if self._sync_requested or self.tx_bytes_available == -1:
                self._tx_active_packet_id = 0

        if sync_requested:
            self._write_data_packet_header(0)
            if timeout:
                self.wait(timeout=timeout, synchronize=False)
            return False

        if not self.is_synchronized:
            self._write_data_packet_header(-1)
            if timeout:
                self.wait(timeout=timeout, synchronize=False)
            return False

        return True


    def synchronize_blocking(self, timeout:float=None) -> bool:
        """Synchronize UART link with other size

        Before data transfer may behind, both sides need to "synchronize".
        This will block until the link is synchronized.

        Arguments:
            timeout: Maximum amount of time to wait for synchronization

        Return:
            Status of the synchronization
        """
        timeout = timeout or 1e9
        start_time = time.time()

        while not self.synchronize():
            if (time.time() - start_time) > timeout:
                return False

            time.sleep(0.100)

        return True


    def read(
        self,
        max_size:int=None,
    ) -> bytes:
        """Read binary data

        Read any data that is immediately available in the RX buffer.

        Arguments:
            max_size: The maximum amount of data to read. If None, the read rx_bytes_available

        Return:
            Read binary data, up to min(rx_bytes_available, max_size)
        """
        if not self.synchronize():
            return bytes()

        # Read everything in the RX buffer is no limit is given
        rx_bytes_available = self.rx_bytes_available
        max_size = max_size or rx_bytes_available
        rx_length = min(max_size, rx_bytes_available)

        if rx_length > 0:
            with self._lock:
                retval = bytes(self._rx_buffer[:rx_length])
                self._rx_buffer = self._rx_buffer[rx_length:]
                logger.debug(f'Read {len(retval)} bytes ({self.rx_bytes_available} still pending)')

            self._write_data_packet_header(0)

        else:
            retval = bytes()

        return retval


    def read_all(
        self,
        length:int=None,
        timeout:float=None,
    ) -> bytes:
        """Read binary data

        Read the specified amount of binary, blocking until all the data is read.

        Arguments:
            length: The amount of data to read
            timeout: The maximum time to wait for the data, if None then block forever until all data is read

        Return:
            Read binary data which may be less than length if the timeout is exceeded
        """
        retval = bytearray()
        start_time = time.time()
        if length and length < 0:
            if not timeout:
                raise ValueError('Must specify timeout if length < 0')
            length = 1e9
        length = length or self.rx_bytes_available

        while length > 0:
            data = self.read(length)
            if len(data) == 0:
                if timeout and (time.time() - start_time) > timeout:
                    return retval
                self.wait(0.010)
                continue

            length -= len(data)
            retval.extend(data)

        return bytes(retval)

    def flush_input(self, timeout:float=0.100):
        """Drop any data in the RX buffer

        This will block for up to timeout while the data is read and dropped
        """
        self.read_all(-1, timeout=timeout)


    def write(
        self,
        data:bytes
    ) -> int:
        """Write binary data

        Write any data that is immediately available to be sent to the other side.
        This will only write up to min(tx_bytes_available, len(data))

        Arguments:
            data: The binary data to write

        Return:
            The number of bytes written. -1 if the link is not synchronized
        """

        if not self.synchronize():
            return -1

        max_tx_bytes = self.tx_bytes_available
        tx_length = min(max_tx_bytes, len(data))

        if tx_length > 0:
            with self._lock:
                logger.debug(f'Writing packet: {tx_length}')
                self._tx_active_packet_id = self._write_data_packet_header(tx_length)
                time.sleep(0.001)
                self._handle.write(data[:tx_length])
                self._handle.flush()

        return tx_length


    def write_all(
        self,
        data:bytes,
        timeout:float=None
    ) -> int:
        """Write binary data

        Write all the given binary data, blocking until all data is written
        or the given timeout is exceeded.

        Arguments:
            data: The binary data to write
            timeout: The maximum amount of time to wait. If None then block forever until all the data is written

        Return:
            The number of bytes written
        """

        start_time = time.time()
        initial_length = len(data)

        while len(data) > 0:
            bytes_written = self.write(data)
            if bytes_written <= 0:
                if timeout and (time.time() - start_time) > timeout:
                    return initial_length - len(data)

                self.wait(0.010)
                continue

            data = data[bytes_written:]

        return initial_length


    def write_command(
        self,
        code:int,
        payload:bytes=None
    ) -> bool:
        """Send a command to the other side

        A "command" consists of an unsigned, 8-bit code and an optional, 6-byte payload.
        While the command is guaranteed to be sent the other side.
        Reception at the other side's application-level is not guaranteed.

        NOTE: Unread commands on the other side will be silently dropped
        """
        if code < 0 or code > 255:
            raise ValueError('code must be uint8')
        if payload:
            if not isinstance(payload, (bytes,bytearray)) or len(payload) > 6:
                raise ValueError('payload must be byte string no longer than 6 bytes')

        if not self.synchronize():
            return False

        with self._lock:
            if not self.is_open:
                raise RuntimeError('Connection not opened')

            header = PacketHeader.create_cmd_header(
                code=code,
                payload=payload
            )
            logger.debug(f'TX {header}')
            self._handle.write(header.serialize())
            self._handle.flush()

        return True


    def read_command(self) -> Command:
        """Read a command sent from the other side

        A "command" consists of an unsigned, 8-bit code and an optional, 6-byte payload.
        If the other side sends commands, then the application should periodically
        call this API to receive the command.

        NOTE: Unread commands will be silently dropped

        If a command is available, then the command.code will be > 0
        """
        if not self.synchronize():
            return Command()

        code = None
        payload = None
        with self._lock:
            if self._rx_cmd_code is not None:
                code = self._rx_cmd_code
                payload = self._rx_cmd_payload
                self._rx_cmd_code = None
                self._rx_cmd_payload = None

        return Command(code, payload)


    def wait(self, timeout:float=None, synchronize=True) -> bool:
        """Wait for a data link event

        This blocks on a Threading.Condition until an event occurs
        on the data link.

        If synchronize=True, then this ensure the data link is synchronized
        """
        with self._lock:
            if synchronize:
                wait_result = None
                start_time = time.time()
                while True:
                    if self.synchronize(timeout=None):
                        if wait_result is not None:
                            return wait_result
                        return self._condition.wait(timeout)

                    if timeout and (time.time() - start_time) > timeout:
                        return False

                    wait_result = self._condition.wait(timeout=.010)

            else:
                return self._condition.wait(timeout)


    def __enter__(self):
        self.open()
        return self

    def __exit__(self, dtype, value, tb):
        self.close()


    def _write_data_packet_header(self, tx_length:int) -> int:
        with self._lock:
            if not self.is_open:
                raise RuntimeError('Connection not opened')

            unused_rx_buffer_length = self._rx_buffer_length - len(self._rx_buffer)

            header = PacketHeader.create_data_header(
                id=self._tx_next_packet_id,
                ack_id=self._rx_previous_data_packet_id,
                tx_length=tx_length,
                rx_available=unused_rx_buffer_length
            )
            self._tx_next_packet_id = max((self._tx_next_packet_id + 1) % 128, 1)

            logger.debug(f'TX {header}')

            self._handle.write(header.serialize())
            self._handle.flush()

        return header.id


    def _rx_loop(self):
        try:
            while not self._rx_thread_active.is_set():
                header = self._parse_packet_header()
                if header is None:
                    continue

                logger.debug(f'RX {header}')
                with self._lock:
                    if header.is_cmd_packet:
                        self._rx_cmd_code = header.cmd_code
                        self._rx_cmd_payload = header.cmd_payload
                        continue

                    if header.data_ack_id == self._tx_active_packet_id:
                        self._tx_active_packet_id = 0

                    if header.data_tx_length == PACKET_REQUEST_SYNCHRONIZATION:
                        header.data_tx_length = 0
                        self._sync_requested = True
                        self._tx_active_packet_id = 0

                    if header.data_tx_length > 0:
                        self._rx_previous_data_packet_id = header.id

                    self._tx_bytes_available = header.data_rx_available
                    self._condition.notify_all()

                    if header.data_tx_length <= 0:
                        continue

                assert self.rx_bytes_available + header.data_tx_length <= self._rx_buffer_length, 'RX buffer overflow'
                self._read_packet_data(header.data_tx_length)

        except:
            if self.is_open:
                raise
        finally:
            self._rx_thread_active.set()


    def _parse_packet_header(self) -> PacketHeader:
        packet_buffer = bytearray()

        while not self._rx_thread_active.is_set():
            if self._handle.in_waiting == 0:
                time.sleep(0.005)
                continue
            c = self._handle.read(1)[0]

            packet_buffer.append(c)
            if len(packet_buffer) < PACKET_HEADER_LENGTH:
                continue

            header = PacketHeader.deserialize(packet_buffer)
            if header:
                return header

            packet_buffer = packet_buffer[1:]
            continue

        return None


    def _read_packet_data(self, packet_length:int):
        activity_timestamp = time.time()
        packet_data = bytearray()

        while len(packet_data) < packet_length and not self._rx_thread_active.is_set():
            max_read_length = self._handle.in_waiting
            if max_read_length == 0:
                time.sleep(0.005)
                continue
            elif (time.time() - activity_timestamp) > 10.0:
                logger.warning(f'Timed-out waiting for packet of length {packet_length}')
                return

            chunk_length = min(max_read_length, packet_length - len(packet_data))
            data = self._handle.read(chunk_length)
            if data:
                activity_timestamp = time.time()
                packet_data.extend(data)

        with self._lock:
            #logger.debug(binascii.hexlify(bytearray(packet_data)))
            self._rx_buffer.extend(packet_data)
            self._write_data_packet_header(0)
            self._condition.notify_all()


class Command(NamedTuple):
    code:int = -1
    payload:bytes = None





PACKET_HEADER_LENGTH = 12
PACKET_DELIMITER1:bytes = b'\xDE\xAD'
PACKET_DELIMITER2:bytes = b'\xBE\xEF'
PACKET_REQUEST_SYNCHRONIZATION:int = -1
PACKET_ACKNOWLEDGE_SYNCHRONIZATION:int = 0
COMMAND_PAYLOAD_LENGTH:int = 6


@dataclass
class PacketHeader:
    id:int = None
    data_ack_id:int = None
    data_tx_length:int = None
    data_rx_available:int = None
    cmd_code:int = None
    cmd_payload:bytearray = None

    @property
    def is_data_packet(self) -> bool:
        return self.id > 0

    @property
    def is_cmd_packet(self) -> bool:
        return self.id < 0

    @staticmethod
    def create_data_header(
        id:int, # pylint: disable=redefined-builtin
        ack_id:int,
        tx_length:int,
        rx_available:int
    ) -> PacketHeader:
        return PacketHeader(
            id=id,
            data_ack_id=ack_id,
            data_tx_length=tx_length,
            data_rx_available=rx_available
        )

    @staticmethod
    def create_cmd_header(
        code:int,
        payload:bytes=None
    ) -> PacketHeader:
        payload = bytearray(payload) if payload else bytearray()
        while len(payload) < COMMAND_PAYLOAD_LENGTH:
            payload.append(0)

        return PacketHeader(
            id=-1,
            cmd_code=code,
            cmd_payload=payload
        )


    @staticmethod
    def deserialize(data:bytes) -> PacketHeader:
        if not (
            len(data) == PACKET_HEADER_LENGTH and
            data[0] == PACKET_DELIMITER1[0] and
            data[1] == PACKET_DELIMITER1[1] and
            data[2] != 0 and
            data[10] == PACKET_DELIMITER2[0] and
            data[11] == PACKET_DELIMITER2[1]
        ):
            return None

        id = struct.unpack('<b', data[2:3])[0]
        header = PacketHeader(id=id)

        if header.is_cmd_packet:
            header.cmd_code = data[3]
            header.cmd_payload = bytearray()
            for c in data[4:10]:
                header.cmd_payload.append(c)

        else:
            header.data_ack_id, header.data_tx_length, header.data_rx_available = struct.unpack('<bhl', data[3:10])
            if header.data_rx_available < 0 or header.data_tx_length < -1:
                return None

        return header


    def serialize(self) -> bytes:
        if not self.id:
            raise RuntimeError('Invalid packet id')

        if self.is_data_packet:
            return struct.pack(
                '<BBbbhlBB',
                PACKET_DELIMITER1[0],
                PACKET_DELIMITER1[1],
                self.id,
                self.data_ack_id,
                self.data_tx_length,
                self.data_rx_available,
                PACKET_DELIMITER2[0],
                PACKET_DELIMITER2[1]
            )
        else:
            return struct.pack(
                '<BBbBBBBBBBBB',
                PACKET_DELIMITER1[0],
                PACKET_DELIMITER1[1],
                -1,
                self.cmd_code,
                *self.cmd_payload,
                PACKET_DELIMITER2[0],
                PACKET_DELIMITER2[1]
            )


    def __str__(self) -> str:
        if not self.id:
            return 'Invalid packet header'

        if self.is_data_packet:
            return f'Data packet: id={self.id} ack={self.data_ack_id} tx_len={self.data_tx_length} rx_len={self.data_rx_available}'
        else:
            return f'Cmd packet: code={self.cmd_code} payload={binascii.hexlify(self.cmd_payload)}'