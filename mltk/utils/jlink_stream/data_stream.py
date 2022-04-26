
import time
from threading import Event, RLock

from mltk.utils import hexdump

from .device_interface import DeviceInterface, MAX_BUFFER_SIZE


WAIT_FOREVER = 4294967.0


class JLinkDataStream(object):
    """JLink data stream"""

    def __init__(
        self, 
        name:str, 
        mode:str, 
        ifc: DeviceInterface, 
        stream_context: dict
    ):
        self._name = name 
        self._mode = mode
        
        self._ifc = ifc
        self._context = stream_context
        
        self._is_opened = Event()
        self._buffer = bytearray()
        self._buffer_lock = RLock()
        self._buffer_event = Event()
        self._notify_event = None
        self._max_read_size = -1
        self._timeout = -1
        self._end_time = -1
        self._requires_processing = False
        self._id_mask = (1 << stream_context['id'])
        
        self._is_opened.set()
        

    @property
    def name(self) -> str:
        """The name of the opened stream"""
        return self._name 
    

    @property
    def mode(self) -> str:
        """The mode the for which the stream was opened, r or w"""
        return self._mode


    @property
    def is_opened(self) -> bool:
        """If the stream is opened to the device"""
        return self._is_opened.is_set() 
    

    @property
    def max_read_size(self) -> int:
        """The maximum amount of data to read
        
        Set to -1 to disable limit
        After each read, this value will decrement by the amount of data read.
        One this value reaches zero, it must be reset otherwise subsequent reads
        will always return zero.
        """
        return self._max_read_size 
    @max_read_size.setter
    def max_read_size(self, val:int):
        if val is None:
            val = -1
        self._max_read_size = val 
    

    @property
    def timeout(self) -> float:
        """The maximum about of time in seconds to read or write data. 
        This is only used if the 'timeout' argument to the read() or write() APIs is None
        Set to -1 to never timeout
        """
        return self._timeout 
    @timeout.setter
    def timeout(self, val: float):
        if val is None:
            val = -1
        self._timeout = val 


    @property
    def end_time(self) -> float:
        """The absolute time in seconds to timeout reading or writing
        
        Set to None to disable.
        If end_time > time.time(), then return from the read() or write() API
        """
        return self._end_time
    @end_time.setter
    def end_time(self, val:float):
        if val is None:
            val = -1
        self._end_time = val 
    

    @property
    def buffer_used(self) -> int:
        """The amount of the device data buffer used
        
        If the stream was opened for reading then this
        is the amount of data that was previous received from
        the device and is waiting to be read by the python script.

        If the stream was opened for writing, then this is
        the amount of data that was previously written and is 
        pending to be sent to the device.
        """
        with self._buffer_lock:
            retval = len(self._buffer)

        return retval
    @property
    def buffer_unused(self) -> int:
        """The amount of the device data buffer that is available"""
        with self._buffer_lock:
            retval = MAX_BUFFER_SIZE - len(self._buffer)

        return retval


    @property
    def read_data_available(self) -> int:
        """The amount of data that is ready to be read by the python script"""
        return self.buffer_used
    @property
    def write_data_available(self) -> int:
        """The amount of data that can immediately be written"""
        return self.buffer_unused
    
    @property
    def buffer_hexdump(self, length=64) -> str:
        """Return a hexdump string"""
        length = min(length, self.buffer_used)
        return hexdump.hexdump(self._buffer[:length], result='return')


    def close(self):
        """Close the data stream with the device"""
        if self._is_opened.is_set():
            self._is_opened.clear()
            self._buffer_event.set()
            
            self._ifc.close(self._name)
    
    
    def read(self, max_size:int = None, timeout:float=None) -> bytes:
        """Read data from data stream opened for reading
        
        NOTE: The only returns the data that is immediately available.
        The amount of data returned may be less than max_size.
        """
        if self.mode != 'r':
            raise Exception(f'Stream: {self.name} not opened for reading')

        timeout = self._get_timeout(timeout)
        max_size = self._get_max_size(max_size)

        start_time = time.time()
        while True:
            self._buffer_event.clear()
            
            if not self.is_opened:
                raise Exception(f'Stream: {self.name} closed') 
            if max_size == 0:
                return None
            
            bufsize = min(self.read_data_available, max_size)
            
            if bufsize > 0:
                retval = self._consume_buffer(bufsize)
                self._notify_event.set()
                return bytes(retval)
            
            elapsed = (time.time() - start_time)
            if elapsed >= timeout:
                return None
            
            if self._end_time > 0:
                time_remaining = self._end_time - time.time()
                if time_remaining <= 0:
                    return None
            else:
                time_remaining = WAIT_FOREVER
            
            self._buffer_event.wait(min(min(timeout - elapsed, time_remaining), 0.100))


    def read_all(self, amount:int, timeout:float=None, initial_timeout:float=None, throw_exception=True) -> bytes:
        """The the specified amount of data"""

        if initial_timeout is None:
            initial_timeout = timeout
        retval = bytearray()
        remaining = amount
        while remaining > 0:
            chunk_timeout = initial_timeout if len(retval) == 0 else timeout
            chunk = self.read(max_size=remaining, timeout=chunk_timeout)
            if chunk is None:
                break 
            remaining -= len(chunk)
            retval.extend(chunk)

        if len(retval) != amount and throw_exception:
            raise Exception('Failed to read all data')

        return bytes(retval)
        
        
    
    def write(self, data:bytes, timeout:float=None, flush=False) -> int:
        """Write data to a data stream opened for writing"""

        if self.mode != 'w':
            raise Exception(f'Stream: {self.name} not opened for writing')
    
        timeout = self._get_timeout(timeout)
        total_write_len = 0
        start_time = time.time()
        
        while len(data) > 0:
            self._buffer_event.clear()
            
            if not self.is_opened:
                raise Exception(f'Stream: {self.name} closed') 

            bufsize = min(self.write_data_available, len(data))
            
            if bufsize > 0:
                self._populate_buffer(data[:bufsize])
                data = data[bufsize:]
                total_write_len += bufsize
                self._requires_processing = True
                self._notify_event.set()
                if len(data) == 0:
                    break
                
            elapsed = (time.time() - start_time)
            if elapsed >= timeout:
                break
            
            if self._end_time > 0:
                time_remaining = self._end_time - time.time()
                if time_remaining <= 0:
                    break
            else:
                time_remaining = WAIT_FOREVER
                
            self._buffer_event.wait(min(min(timeout - elapsed, time_remaining), 0.100))
        
        
        if flush: 
            self.flush(timeout=timeout)
        
        
        return total_write_len
    
    
    def flush(self, timeout:float=None):
        """Wait while any pending data is transferred to/from the device"""
        timeout = self._get_timeout(timeout)
        start_time = time.time()
        while self.buffer_used > 0:
            self._buffer_event.clear()
            
            if not self.is_opened:
                raise Exception(f'Stream: {self.name} closed') 
            
            elapsed = (time.time() - start_time)
            if elapsed >= timeout:
                raise Exception('Time-out waiting for buffer to flush')
            
            if self._end_time > 0:
                time_remaining = self._end_time - time.time()
                if time_remaining <= 0:
                    break
            else:
                time_remaining = WAIT_FOREVER
            
            self._buffer_event.wait(min(min(timeout - elapsed, time_remaining), 0.100))
            
            
    

    def _set_notify_event(self, event):
        self._notify_event = event 
     
    
    def _process(self, buffer_status_mask):
        if not self._requires_processing and (buffer_status_mask & self._id_mask) == 0:
            return
        
        self._requires_processing = False
        
        if self.mode == 'r':
            max_read_len = self.buffer_unused
            if max_read_len > 0:
                data = self._ifc.read(self._context, max_read_len)
                if data:
                    self._populate_buffer(data)
                
            else:
                self._requires_processing = True
            
        elif self.mode == 'w':
            write_len = self._ifc.write(self._context, self._buffer)
            if write_len:
                self._consume_buffer(write_len)
            
            if self.buffer_used > 0:
                self._requires_processing = True

    
    
    
    def _consume_buffer(self, size) -> bytes:
        with self._buffer_lock:
            retval = self._buffer[:size]
            self._buffer = self._buffer[size:]
            
            if self._max_read_size != -1:
                if size <= self._max_read_size:
                    self._max_read_size -= size 
                else:
                    self._max_read_size = 0
            
        if self.mode == 'w':
            self._buffer_event.set()
        
        return retval
    
    
    def _populate_buffer(self, data):
        with self._buffer_lock:
            if isinstance(data, str):
                data = data.encode()
            self._buffer.extend(data)

        if self.mode == 'r':
            self._buffer_event.set()


    def _get_timeout(self, timeout:float) -> float:
        if timeout is None:
            timeout = self._timeout
        if timeout == -1:
            timeout = WAIT_FOREVER
        return timeout

    
    def _get_max_size(self, max_size:int) -> int:
        if max_size is None:
            max_size = self._max_read_size
        if max_size == -1:
            max_size = MAX_BUFFER_SIZE
        return max_size
    

    def __iter__(self):
        return self

    def __next__(self):
        retval = self.read()
        if retval is None:
            raise StopIteration  # Done iterating.
        
        return retval


    def __enter__(self):
        return self


    def __exit__(self, dtype, value, traceback):
        self.close()

