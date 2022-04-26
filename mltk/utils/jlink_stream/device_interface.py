
import time
import struct
from threading import RLock

from mltk.utils.jlink import JLink
from mltk.utils.python import prepend_exception_msg


# See <mltk root>/cpp/shared/jlink_stream/jlink_stream/jlink_stream_interface.hpp


CONTEXT_MAGIC_NUMBER            = 0xF3f37BA2
BUFFER_MAGIC_NUMBER             = 0xE3D37B32


COMMAND_TIMEOUT                 = 15.0
MAX_BUFFER_SIZE                 = 8*1024

STATUS_INIT_REQUIRED            = 0
STATUS_INIT_FAILED              = 1
STATUS_INVOKE_INIT              = ~CONTEXT_MAGIC_NUMBER
STATUS_IDLE                     = 2
STATUS_READY                    = 3
STATUS_EXECUTING                = 4
STATUS_COMPLETE                 = 5


COMMAND_OPEN_FOR_READ           = 1
COMMAND_OPEN_FOR_WRITE          = 2
COMMAND_CLOSE                   = 3
COMMAND_READ_BUFFER_STATUS_MASK = 4
COMMAND_READ_BUFFER             = 5
COMMAND_WRITE_BUFFER            = 6

RESULT_SUCCESS                  = 0
RESULT_ERROR                    = 1
RESULT_UNKNOWN_CMD              = 2
RESULT_NOT_FOUND                = 3
RESULT_READ_ONLY                = 4
RESULT_WRITE_ONLY               = 5
RESULT_BAD_ARGS                 = 6
RESULT_MALLOC_FAILED            = 7
RESULT_ALREADY_OPENED           = 8
RESULT_NOT_OPENED               = 9

RESULT_STR = {}
RESULT_STR[RESULT_SUCCESS]          = 'SUCCESS'
RESULT_STR[RESULT_ERROR]            = 'ERROR'
RESULT_STR[RESULT_UNKNOWN_CMD]      = 'UNKNOWN_CMD'
RESULT_STR[RESULT_NOT_FOUND]        = 'NOT_FOUND'
RESULT_STR[RESULT_READ_ONLY]        = 'READ_ONLY'
RESULT_STR[RESULT_WRITE_ONLY]       = 'WRITE_ONLY'
RESULT_STR[RESULT_BAD_ARGS]         = 'BAD_ARGS'
RESULT_STR[RESULT_MALLOC_FAILED]    = 'MALLOC_FAILED'
RESULT_STR[RESULT_ALREADY_OPENED]   = 'ALREADY_OPENED'
RESULT_STR[RESULT_NOT_OPENED]       = 'NOT_OPENED'


CONTEXT_OFFSET_MAGIC_NUMBER     = 0
CONTEXT_OFFSET_STATUS           = 1
CONTEXT_OFFSET_TRIGGER_ADDR     = 2
CONTEXT_OFFSET_TRIGGER_VAL      = 3
CONTEXT_OFFSET_CMD_CODE         = 4
CONTEXT_OFFSET_CMD_RESULT       = 5
CONTEXT_OFFSET_CMD_LENGTH       = 6
CONTEXT_OFFSET_CMD_BUFFER_ADDR  = 7
CONTEXT_OFFSET_COUNT            = 8

BUFFER_OFFSET_MAGIC_NUMBER      = 0
BUFFER_OFFSET_ID                = 1
BUFFER_OFFSET_START             = 2
BUFFER_OFFSET_END               = 3
BUFFER_OFFSET_HEAD              = 4
BUFFER_OFFSET_TAIL              = 5
BUFFER_OFFSET_LENGTH            = 6
BUFFER_OFFSET_COUNT             = 7


class DeviceInterface(object):
    
    def __init__(self, options): 
        self._jlink = JLink(library_paths=options.lib_path)
        self._jlink.set_interface(options.interface)
        self._jlink.set_speed(options.clock)
        self._jlink.set_core(options.core)
        self._sram_base_address = options.sram_address 
        self._sram_size         = options.sram_size 

        self._lock = RLock()
        self._cmd_lock = RLock()
        self._context_address = 0
        self._trigger_address = 0
        self._trigger_value = 0
        self._command_buffer_address = 0
        

    @property
    def buffer_status_mask(self) -> int:
        data = self._issue_command(COMMAND_READ_BUFFER_STATUS_MASK)
        return struct.unpack('<L', data)[0]
    

    @property
    def is_connected(self) -> bool:
        return self._jlink.is_connected()
    
    
    def open(self, name:str, mode:str):
        # pylint: disable=import-outside-toplevel
        from .data_stream import JLinkDataStream 
        
        if not self.is_connected:
            raise Exception('Not connected')
        
        if mode == 'r':
            cmd = COMMAND_OPEN_FOR_READ
        elif mode == 'w':
            cmd = COMMAND_OPEN_FOR_WRITE
        else:
            raise Exception(f'Unsupported mode: {mode}')
        
        # Ensure the connection is closed
        self._issue_command(COMMAND_CLOSE, name, ignore_error=True)
        
        res = self._issue_command(cmd, name)
        
        stream_context = {}
        stream_context['id'], stream_context['base_address'] = struct.unpack('<LL', res)
        
        stream = JLinkDataStream(
            name=name, 
            mode=mode, 
            ifc=self, 
            stream_context=stream_context
        )
        
        return stream
    

    def close(self, name:str):
        if not self.is_connected:
            raise Exception('Not connected')
    
        self._issue_command(COMMAND_CLOSE, name)
    

    def read(self, context:dict, length:int) -> bytes:
        if length == 0:
            return None

        if not self.is_connected:
            raise Exception('Not connected')
        
        buffer_context = self._read_buffer_context(context['base_address'])
        
        read_length = min(length, buffer_context['length'])
        if read_length == 0:
            return None
        
        length_to_end = buffer_context['end'] - buffer_context['head']
        data = self._jlink_read8(buffer_context['head'], min(length_to_end, read_length))

        if read_length > length_to_end:
            data2 = self._jlink_read8(buffer_context['start'], read_length - length_to_end)
            data.extend(data2)
            
        cmd_data = struct.pack('<LL', buffer_context['id'], read_length)
        self._issue_command(COMMAND_READ_BUFFER, cmd_data)
        
        return bytes(data)

    
    def write(self, context:dict, data:bytes) -> int:
        if data == None or len(data) == 0:
            return
        
        if not self.is_connected:
            raise Exception('Not connected')
        
        
        buffer_context = self._read_buffer_context(context['base_address'])
        
        device_buffer_available = MAX_BUFFER_SIZE - buffer_context['length']
        write_length = min(len(data), device_buffer_available)
        if not write_length:
            return
        
        length_to_end = buffer_context['end'] - buffer_context['tail']
        
        chunk_length = min(length_to_end, write_length)
        self._jlink_write8(buffer_context['tail'], data[:chunk_length])
        
        if write_length > length_to_end:
            self._jlink_write8(buffer_context['start'], data[chunk_length:chunk_length + (write_length-length_to_end)])
            
        cmd_data = struct.pack('<LL', buffer_context['id'], write_length)
        self._issue_command(COMMAND_WRITE_BUFFER, cmd_data)
        
        return write_length

    
    def connect(self, reset_device=False):
        # Open JLink connection
        try:
            self._jlink.connect()
        except Exception as e:
            raise type(e)(f"Failed to open JLINK connection: {e}")
        
        # Reset the device if necessary
        if reset_device:
            try:
                self._jlink.reset()
                self._jlink.resume()
                time.sleep(0.250) # Wait a moment for the device to restart
            except Exception as e:
                self.disconnect()
                raise type(e)(f'Failed to reset target: {e}')

        try:
            # Find the context's address in the device's SRAM
            self._context_address = self._find_context_base_address()
            
            # Read the contents of the context
            context = self._read_context()
            
            # Save the context constant values
            self._trigger_address           = context['trigger_address']
            self._trigger_value             = context['trigger_value']
            self._command_buffer_address   = context['command_buffer_address']
            
            # If the context needs to be initialized
            if context['status'] == STATUS_INIT_REQUIRED:
                context = self._initialize_context()
        except Exception as e:
            self.disconnect()
            prepend_exception_msg(e, 'Failed read context')
            raise
    
    
    def disconnect(self):
        with self._lock:
            try:
                self._jlink.close()
            except: 
                pass


    def _find_context_base_address(self) -> int:
        start_time = time.time()

        def _query_base_address(base_address):
            value = self._jlink_read32(base_address)
            
            # If a value within the SRAM address range is found, 
            # Then query this address value, if it's a pointer
            # To the context, then the context should have its MAGIC_NUMBER set
            if value > self._sram_base_address and value < base_address:
                # Query the value at this address
                value2 =  self._jlink_read32(value)
                
                # If the queried value is the magic number 
                # then the context address has been found
                if value2 == CONTEXT_MAGIC_NUMBER:
                    return value
                
            return None
                
                
        while (time.time() - start_time) < 30.0:
            # The the SRAM size has been specified
            if self._sram_size != -1:
                base_address = self._sram_base_address + self._sram_size - 4
            
                context_address = _query_base_address(base_address)
                if context_address:
                    return context_address
                else:
                    time.sleep(0.50)
                    
            else:
                # The SRAM size isn't known, so iterate through the various RAM sizes
                for ram_size in range(96, 1025, 32):
                    base_address = self._sram_base_address + ram_size*1024 - 4
                    context_address = _query_base_address(base_address)
                    if context_address:
                        return context_address
            
                time.sleep(0.50)
        
        raise Exception("Timed out searching for context's base address. Ensure the device is executing properly")
    
    
    def _initialize_context(self) -> dict:
        context = self._read_context()

        # Write the initialization status to the context
        self._write_context(status=STATUS_INVOKE_INIT)
    
        # Wait until the status becomes idle
        start_time = time.time()
        while True:
            if (time.time() - start_time) > 30.0:
                raise Exception("Timed out waiting to initialize context")
            context = self._read_context()
            if context['status'] == STATUS_IDLE:
                break
            elif context['status'] == STATUS_INIT_FAILED:
                raise Exception("Failed to initialize context")
            
        return context
        
        
    def _read_context(self) -> dict:
        data = self._jlink_read8(self._context_address, CONTEXT_OFFSET_COUNT*4)
        
        context = {}
        context['magic_number']             = self._unpack_data(data, CONTEXT_OFFSET_MAGIC_NUMBER)
        context['status']                   = self._unpack_data(data, CONTEXT_OFFSET_STATUS)
        context['trigger_address']          = self._unpack_data(data, CONTEXT_OFFSET_TRIGGER_ADDR)
        context['trigger_value']            = self._unpack_data(data, CONTEXT_OFFSET_TRIGGER_VAL)
        context['command_code']             = self._unpack_data(data, CONTEXT_OFFSET_CMD_CODE)
        context['command_result']           = self._unpack_data(data, CONTEXT_OFFSET_CMD_RESULT)
        context['command_length']           = self._unpack_data(data, CONTEXT_OFFSET_CMD_LENGTH)
        context['command_buffer_address']   = self._unpack_data(data, CONTEXT_OFFSET_CMD_BUFFER_ADDR)
        
        if context['magic_number'] != CONTEXT_MAGIC_NUMBER:
            raise Exception("Context does not contain a valid magic number")

        return context
    
    
    def _write_context(self, status:int, cmd:int=None, data=None):
        self._lock.acquire()
        
        try:
            self._jlink.halt()
            
            try:
                if cmd:
                    # Write the 'command_code' field
                    self._jlink.write_mem32(self._context_address + CONTEXT_OFFSET_CMD_CODE*4, cmd)
                
                if data:
                    # Write the 'command_length' field
                    self._jlink.write_mem32(self._context_address + CONTEXT_OFFSET_CMD_LENGTH*4, len(data))
                    
                    # Populate the 'command_buffer'
                    if isinstance(data, str):
                        data = data.encode()
                    cmd_buffer = bytes(data)
                    self._jlink.write_mem8(self._command_buffer_address, cmd_buffer)
                    
                
                # Write the 'status' field
                self._jlink.write_mem32(self._context_address + CONTEXT_OFFSET_STATUS*4, status)
                
                # Write the 'trigger' value to invoke the Kernel to process the context
                self._jlink.write_mem32(self._trigger_address, self._trigger_value)
                
            finally:
                self._jlink.resume() 
            
        finally:
            self._lock.release()
    
    
    def _issue_command(self, cmd, data=None, ignore_error=False):
        self._cmd_lock.acquire()
        
        try:
            context = self._read_context()
            
            if context['status'] == STATUS_EXECUTING:
                raise Exception('Device is currently executing another command')
            
            # Write the command info to the context
            self._write_context(STATUS_READY, cmd, data)
            
            # Wait for the command to complete
            start_time = time.time()
            while (time.time() - start_time) < COMMAND_TIMEOUT:
                context = self._read_context()
                
                if context['status'] == STATUS_COMPLETE:
                    break
                
                time.sleep(0.005)
            
            if context['status'] != STATUS_COMPLETE:
                raise Exception("Timed-out waiting  for device to execute command")
            
            response_data = None
            if context['command_length'] > 0:
                try:
                    response_data = self._jlink.read_mem8(self._command_buffer_address, context['command_length'])
                except Exception as e:
                    raise type(e)('Failed to read response data, err:%s' % e)
                
            if context['command_result'] != RESULT_SUCCESS and not ignore_error:
                if response_data:
                    response_data = response_data.decode('utf-8')
                else:
                    response_data = ''
                    
                if context['command_result'] < len(RESULT_STR):
                    desc = ' - %s' % RESULT_STR[context['command_result']]
                else:
                    desc = ''
                    
                msg = 'Command failed (%d%s)' % (context['command_result'], desc)
                
                if len(response_data) > 0:
                    msg += ', err: %s' % response_data
                    
                raise Exception(msg)
            
            return response_data
        
        finally:
            self._cmd_lock.release()
    
    
    def _read_buffer_context(self, base_address) -> dict:
        data = self._jlink_read8(base_address, BUFFER_OFFSET_COUNT*4)

        context = {}
        context['magic_number'] = self._unpack_data(data, BUFFER_OFFSET_MAGIC_NUMBER)
        context['id']           = self._unpack_data(data, BUFFER_OFFSET_ID)
        context['start']        = self._unpack_data(data, BUFFER_OFFSET_START)
        context['end']          = self._unpack_data(data, BUFFER_OFFSET_END)
        context['head']         = self._unpack_data(data, BUFFER_OFFSET_HEAD)
        context['tail']         = self._unpack_data(data, BUFFER_OFFSET_TAIL)
        context['length']       = self._unpack_data(data, BUFFER_OFFSET_LENGTH)
        
        if context['magic_number'] != BUFFER_MAGIC_NUMBER:
            raise Exception('Stream buffer context is not valid on device')

        return context


    def _unpack_data(self, data, offset) -> int:
        start_index = offset*4
        end_index = start_index + 4
        return struct.unpack('<L', data[start_index:end_index])[0]
    

    def _jlink_read8(self, addr, length) -> bytearray:
        with self._lock:
            return bytearray(self._jlink.read_mem8(addr, length))
    

    def _jlink_read32(self, addr) -> int:
        with self._lock:
            return self._jlink.read_mem32(addr)
    
    
    def _jlink_write8(self, addr, data):
        with self._lock:
            return self._jlink.write_mem8(addr, data)


    def _jlink_write32(self, addr, value):
        with self._lock:
            return self._jlink.write_mem32(addr, value)
