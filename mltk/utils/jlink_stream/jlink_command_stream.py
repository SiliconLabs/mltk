import struct
from .jlink_stream import JlinkStream, JlinkStreamOptions
from .data_stream import JLinkDataStream


class JlinkCommandStream:
    """Helper class for issuing a command/response to embedded device via J-Link"""
    def __init__(
        self,
        command_stream='cmd',
        response_stream='res',
        options:JlinkStreamOptions = None,
    ):
        self._jlink_ifc = JlinkStream(options=options)
        self._command_stream_name = command_stream
        self._response_stream_name = response_stream
        self._command_stream:JLinkDataStream = None
        self._response_stream:JLinkDataStream = None


    def open(self, reset_device=False):
        """Open the JLink connection and command streams"""
        self._jlink_ifc.connect(reset_device=reset_device)

        try:
            self._command_stream = self._jlink_ifc.open(
                name=self._command_stream_name,
                mode='w'
            )

            self._response_stream = self._jlink_ifc.open(
                name=self._response_stream_name,
                mode='r'
            )
        except:
            self._jlink_ifc.disconnect()


    def close(self):
        """Close the JLink connection and streams"""
        try:
            self._jlink_ifc.disconnect()
        finally:
            self._command_stream = None
            self._response_stream = None


    def issue(self, data:bytes, timeout:float=7.0, no_response=False) -> bytes:
        """Send a command to the device and receive the command response"""
        if self._command_stream is None:
            raise Exception('Not connected')

        cmd_length = len(data)
        cmd_data = bytearray(struct.pack('<L', cmd_length))
        cmd_data.extend(data)

        self._command_stream.write(cmd_data, timeout=timeout, flush=True)

        if no_response:
            return None

        length_bytes = self._response_stream.read(4, timeout=timeout)
        if length_bytes is None or len(length_bytes) != 4:
            raise TimeoutError('Timed-out waiting for response')
        length = struct.unpack('<L', length_bytes)[0]
        return self._response_stream.read_all(length, timeout=timeout)



    def __enter__(self):
        self.open()
        return self

    def __exit__(self, dtype, value, tb):
        self.close()
