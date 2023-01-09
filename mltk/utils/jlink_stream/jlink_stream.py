
import traceback
from threading import Thread, Event, RLock


from .data_stream import JLinkDataStream


class JlinkStreamOptions:
    """JLinkStream configuration options"""
    serial_number = None
    core = 'cortex-m4'
    interface = 'swd'
    clock = 12000
    sram_address = 0x20000000
    sram_size = -1
    lib_path = None
    polling_period = 0.1




class JlinkStream:
    """This allows for transferring binary data between a Python script and a JLink-enabled embedded device via the debug interface

    See the source code on Github: `mltk/utils/jlink_stream <https://github.com/siliconlabs/mltk/blob/master/mltk/utils/jlink_stream>`_
    """
    def __init__(self, options:JlinkStreamOptions = None):
        # import here to avoid circular import
        from .device_interface import DeviceInterface # #pylint: disable=import-outside-toplevel
        if options is None:
            options = self.default_options

        self._ifc = DeviceInterface(options)
        self._is_connected = Event()
        self._data_available = Event()
        self._stream_lock = RLock()

        self._streams = {}

        self.polling_period = options.polling_period


    @property
    def default_options(self) -> JlinkStreamOptions:
        """Default configuration options"""
        return JlinkStreamOptions()

    @property
    def is_connected(self) -> bool:
        """Return if the device is connected"""
        return self._is_connected.is_set()


    def connect(self, threaded=True, reset_device=False):
        """Open a connection to an embedded device via J-Link

        NOTE: The embedded device must be actively running the JLink library
        """
        if self.is_connected:
            raise Exception('Already connected')

        self._ifc.connect(reset_device=reset_device)

        self._is_connected.set()
        self._data_available.clear()
        if threaded:
            t = Thread(
                target=self._processing_thread_loop,
                name='Jlink Stream Data Loop',
                daemon=True
            )
            t.start()


    def disconnect(self):
        """Close the connection to the embedded device"""
        if not self.is_connected:
            return

        self._is_connected.clear()
        self._data_available.set()

        with self._stream_lock:
            for _, stream in self._streams.items():
                try:
                    stream.close()
                except:
                    pass

            self._streams = {}

        try:
            self._ifc.disconnect()
        except:
            pass


    def open(self, name:str, mode='r') -> JLinkDataStream:
        """Open a data stream to the embedded device"""

        with self._stream_lock:
            if name in self._streams:
                raise Exception(f'Stream with name: {name} already opened')

        stream = self._ifc.open(name=name, mode=mode)
        # pylint: disable=protected-access
        stream._set_notify_event(self._data_available)

        with self._stream_lock:
            self._streams[name] = stream

        self._data_available.set()

        return stream


    def close(self, name:str):
        """Close a device data stream"""
        with self._stream_lock:
            if name in self._streams:
                raise Exception(f'Stream with name: {name} not found')

            stream = self._streams[name]
            del self._streams[name]

        stream.close()


    def read(self, name:str, max_size:int=None, timeout:float=None) -> bytes:
        """Read data from a data stream opened for reading"""
        with self._stream_lock:
            if not name in self._streams:
                raise Exception(f'Stream with name: {name} not found')

            stream = self._streams[name]

        return stream.read(max_size, timeout)


    def write(self, name:str, data:bytes, timeout:float=None) -> int:
        """Write data to a data stream opened from writing"""
        with self._stream_lock:
            if not name in self._streams:
                raise Exception(f'Stream with name: {name} not found')

            stream = self._streams[name]

        return stream.write(data, timeout)


    def process(self):
        """Process the device data streams

        This is periodically called in a separated thread if
        'threaded=True' in the connect() API.
        Otherwise, this should be periodically called.

        """
        if not self.is_connected:
            raise Exception('Not connected')

        # Read the buffer mask to immediately see if any more data is being
        buffer_status_mask = self._ifc.buffer_status_mask

        if buffer_status_mask == 0:
            # Wait for data to be available or for the polling period to expire
            self._data_available.wait(timeout=self.polling_period)

        if not self.is_connected:
            return

        # Clear the flag as we're now going to process all the streams
        self._data_available.clear()

        if buffer_status_mask == 0:
            # Read the data available mask from the device
            buffer_status_mask = self._ifc.buffer_status_mask

        # Acquire the streams lock
        with self._stream_lock:
            closed_streams = []

            # Process each opened stream
            for name, stream in self._streams.items():
                # If the stream is closed then add it to the list
                if not stream.is_opened:
                    closed_streams.append(name)
                    continue

                # Process the stream
                try:
                    # pylint: disable=protected-access
                    stream._process(buffer_status_mask)
                except:
                    traceback.print_exc()

            # Remove any closed streams
            for stream in closed_streams:
                del self._streams[stream]



    def _processing_thread_loop(self):
        """Data stream processing loop

        This is periodically called in the Python thread.
        It polls the embedded device's data streams
        """
        # While the interface is active
        while self.is_connected:
            self.process()


    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, dtype, value, tb):
        self.disconnect()

