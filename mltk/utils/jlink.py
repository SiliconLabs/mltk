from typing import List, Union
import sys
import os
import ctypes
import time


from .commander import download_commander

JLINK_CORE_CORTEX_M0        = 0x060000FF
JLINK_CORE_CORTEX_M1        = 0x010000FF
JLINK_CORE_CORTEX_M3        = 0x030000FF
JLINK_CORE_CORTEX_M33       = 0x030000FF
JLINK_CORE_CORTEX_M4        = 0x0E0000FF


JLINK_CORE_IDS = {}
JLINK_CORE_IDS['cortex-m0'] = JLINK_CORE_CORTEX_M0
JLINK_CORE_IDS['cortex-m1'] = JLINK_CORE_CORTEX_M1
JLINK_CORE_IDS['cortex-m3'] = JLINK_CORE_CORTEX_M3
JLINK_CORE_IDS['cortex-m33'] = JLINK_CORE_CORTEX_M33
JLINK_CORE_IDS['cortex-m4'] = JLINK_CORE_CORTEX_M4


class JLinkException(Exception): 
    pass


    
def _CheckErrorDecorator(fn):
    def checked_transaction(self, *args):
        self.dll_clear_error()
        ret = fn(self, *args)
        errno = self.dll_has_error()
        if errno:
            raise JLinkException(f'JLINK error code: {errno}')
        return ret
    return checked_transaction



class JLink(object):
    """JLink interface to an embedded device"""
    # See https://github.com/markrages/jlinkpy/tree/master/jlink
    #     https://github.com/deadsy/pycs/blob/master/jlink.py
    

    def __init__(
        self, 
        library_paths:List[str]=None
    ):
        try:
            self.jl, self.jlink_lib_name = _get_jlink_dll(library_paths)
        except Exception as e:
            raise JLinkException(f'Failed to load Jlink library, err: {e}')

        self.dll_open()
        self._core = None
        self._device = None
        self._speed = None
        
    def connect(self):
        """Connect to a device via JLINK"""

        if not self.dll_is_connected():
            if self._device:
                self.set_device(self._device)
            elif self._core:
                self.set_core(self._core)
                    
            if self._speed: 
                self.set_speed(self._speed)
                    
            self.dll_connect()


    def close(self):
        """Close the connection to the device"""
        try:
            self.dll_close()
        except: 
            pass
    
    def set_log_file_path(self, path:str):
        self.dll_set_log_path(path)


    def set_interface(self, interface:str):
        if interface == 'jtag':
            return self.dll_tif_select(0)
        elif interface == 'swd':
            return self.dll_tif_select(1)
        else: 
            raise JLinkException(f'Interface not supported: {interface}')
        
        
    def is_connected(self) -> bool:
        return self.dll_is_connected() == 1
    

    def reset(self):
        return self.dll_reset()
    

    def halt(self):
        self.dll_halt()
    
    
    def is_halted(self) -> bool:
        retval = self.dll_is_halted()
        return retval == 1
    
    
    def resume(self, timeout=0):
        self.dll_go()
        
        if timeout > 0:
            start_time = time.time()
            while self.is_halted():
                self.dll_go()
                
                if (time.time() -  start_time) > timeout:
                    break
                time.sleep(0.005)
    
    
    def get_jtag_id(self) -> int:
        buf, _ = self.dll_get_id()
        return self._cast_buffer(buf, ctypes.c_uint32, 1)
    

    def set_speed(self, speed_khz: int):
        self.dll_set_speed(speed_khz)
        self._speed = speed_khz
    
    
    def set_core(self, core_id_str:str):
        try:
            self.dll_execute_command('device ' + core_id_str)
            self._core = core_id_str
        except Exception as e:
            raise JLinkException(f'Failed to set core to: {core_id_str}, error: {e}')
        
        
    def set_device(self, device_str):
        try:
            self.dll_execute_command('device ' + device_str)
            self._device = device_str
        except Exception as e:
            raise JLinkException(f'Failed to set device to: {device_str}, error: {e}')
        
    
    def issue_command(self, cmd:str):
        try:
            return self.dll_execute_command(cmd)
        except Exception as e:
            raise JLinkException(f'Failed to execute command: {cmd}, error: {e}')
    
    
    def read_reg(self, reg: Union[str,int]) -> int:
        return self.dll_read_reg(self._convert_reg(reg))
    
    
    def write_reg(self, reg:Union[str, int], value:int):
        return self.dll_write_reg(self._convert_reg(reg), value)
    
    
    def read_mem32(self, address:int, count=1) -> List[int]:
        buf, _ = self.dll_read_mem_u32(address, count)
        return self._cast_buffer(buf, ctypes.c_uint32, count)
    

    def read_mem8(self, address:int, length:int) -> bytes:
        buf, _ = self.dll_read_mem(address, length)
        a = ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8))
        retval = bytearray()

        for i in range(length):
            retval.append(a[i])
        
        return bytes(retval)
    
    
    def write_mem8(self, address:int, values: bytes):
        self.dll_write_mem(address, values)
    
    
    def write_mem32(self, address:int, value: List[int]):
        if not isinstance(value, (list,tuple)):
            if not isinstance(value, ctypes.c_ulong):
                value = ctypes.c_ulong(value)
            self.dll_write_u32(address, value)
        else:
            for x in value:
                if not isinstance(x, ctypes.c_ulong):
                    x= ctypes.c_ulong(x)
                self.dll_write_u32(address, x)


    def write_uint32_le(self, address, value):
        if not isinstance(value, (list,tuple)):
            self.dll_write_u32(address, value)
        else:
            for x in value:
                self.dll_write_u32(address, x)

    def load_binary_data(self, file_path:str, dest_address:int, offset=0, length=None):
        with open(file_path, 'rb') as f:
            data = f.read()
            
        if length is None:
            length = len(data)
        data = data[offset:offset+length]
            
        self.write_mem8(dest_address, data)
        
        
    def program_flash(self, bin_data:bytes, address:int):
        self.dll_begin_download()
        self.write_mem8(address, bin_data)
        return self.dll_end_download()
        
    
    def program_file(self, file_path:str, address:int):
        with open(file_path, 'rb') as f:
            return self.program_flash(f.read(), address)

        
    def format_register(self, reg:Union[str,int]) -> str:
        value = self.read_reg(reg)
        if isinstance(reg, str):
            s = reg.upper()
        else:
            if reg == 13:
                s = ' SP'
            elif reg == 14:
                s = ' LR'
            elif reg == 15:
                s = ' PC'
            else:
                s = 'R%2d' % reg
            
        return f'{s}: 0x{value:08X}'
        
        
    def format_registers(self) -> str:
        print("Registers:")
        retval = 'Registers:\n'
        for r in range(16):
            retval += f'{self.format_register(r)}'
        return retval
        

    def _convert_reg(self, reg) -> int:
        if reg == 'pc':
            return 15
        elif reg == 'lr':
            return 14 
        elif reg == 'sp':
            return 13
        else:
            return reg

    def _cast_buffer(self, buf, dtype, count=1, force_array=False):
        a = ctypes.cast(buf, ctypes.POINTER(dtype))
        
        if count == 1 and not force_array:
            return a[0]
            
        l = []
        for i in range(count):
            l.append(a[i])
            
        return l



    def dll_clear_error(self): 
        self.jl.JLINK_ClrError()
    
    def dll_has_error(self) -> int: 
        return self.jl.JLINK_HasError()
    
    @_CheckErrorDecorator 
    def dll_set_log_path(self, path):
        path_str = ctypes.create_string_buffer(path)
        self.jl.JLINKARM_SetLogFile(path_str)

    @_CheckErrorDecorator
    def dll_tif_select(self, tif): 
        return self.jl.JLINKARM_TIF_Select(tif)
    @_CheckErrorDecorator
    def dll_set_speed(self, khz): 
        # void JLINKARM_SetSpeed(long int khz);
        return self.jl.JLINKARM_SetSpeed(khz)
    def dll_get_id(self): 
        # U32 JLINKARM_GetId(void);
        return self.jl.JLINKARM_GetId()
    
    @_CheckErrorDecorator
    def dll_select_device_family(self, device_family_id):
        # void JLINKARM_SelectDeviceFamily (int DeviceFamily)
        return self.jl.JLINKARM_SelectDeviceFamily(device_family_id)
    @_CheckErrorDecorator
    def dll_is_opened(self):
        # char JLINKARM_IsOpen(void);
        return self.jl.JLINKARM_IsOpen()
    @_CheckErrorDecorator
    def dll_is_connected(self):
        # char JLINKARM_IsConnected(void);
        return self.jl.JLINKARM_IsConnected()
    @_CheckErrorDecorator
    def dll_is_halted(self):
        # char JLINKARM_IsHalted(void);
        return self.jl.JLINKARM_IsHalted()
    @_CheckErrorDecorator
    def dll_reset(self): 
        return self.jl.JLINKARM_Reset()
    @_CheckErrorDecorator
    def dll_halt(self): 
        return self.jl.JLINKARM_Halt()
    @_CheckErrorDecorator
    def dll_clear_tck(self): 
        return self.jl.JLINKARM_ClrTCK()
    @_CheckErrorDecorator
    def dll_clear_tms(self): 
        return self.jl.JLINKARM_ClrTMS()
    @_CheckErrorDecorator
    def dll_set_tms(self): 
        return self.jl.JLINKARM_SetTMS()
    @_CheckErrorDecorator
    def dll_read_reg(self,r): 
        return self.jl.JLINKARM_ReadReg(r)
    @_CheckErrorDecorator
    def dll_write_reg(self,r,val): 
        return self.jl.JLINKARM_WriteReg(r,val)
    @_CheckErrorDecorator
    def dll_write_u32(self,r,val): 
        return self.jl.JLINKARM_WriteU32(r,val)
    @_CheckErrorDecorator                       
    def dll_open(self): 
        return self.jl.JLINKARM_Open()
    @_CheckErrorDecorator                       
    def dll_close(self): 
        return self.jl.JLINKARM_Close()
    @_CheckErrorDecorator                       
    def dll_connect(self): 
        # int JLINKARM_Connect(void);
        return self.jl.JLINKARM_Connect()
    @_CheckErrorDecorator      
    def dll_go(self): 
        return self.jl.JLINKARM_Go()
    @_CheckErrorDecorator
    def dll_write_mem(self,startaddress, data):
        buf=ctypes.create_string_buffer(bytes(data))
        return self.jl.JLINKARM_WriteMem(startaddress,len(data),buf)
    @_CheckErrorDecorator
    def dll_read_mem(self, startaddress, length):
        buf=ctypes.create_string_buffer(length)
        ret=self.jl.JLINKARM_ReadMem(startaddress,length, buf)
        return buf, ret
    @_CheckErrorDecorator
    def dll_read_mem_u32(self, startaddress, count=1):
        buftype=ctypes.c_uint32 * int(count)
        buf=buftype()
        ret=self.jl.JLINKARM_ReadMemU32(startaddress, count, buf, 0)
        return buf,ret
    
    @_CheckErrorDecorator
    def dll_execute_command(self, cmd):
        # int JLINKARM_ExecCommand(char *sIn, char *sBuffer, int buffersize);
        fn = self.jl.JLINKARM_ExecCommand
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        command = ctypes.create_string_buffer(cmd.encode('utf-8'))
        result = ctypes.create_string_buffer(128)
        fn(command, result, len(result))

        return result.value

    @_CheckErrorDecorator 
    def dll_begin_download(self):
        self.jl.JLINKARM_BeginDownload(0)
    @_CheckErrorDecorator 
    def dll_end_download(self):
        return self.jl.JLINKARM_EndDownload()
        
#     @_CheckErrorDecorator
#     def _download_file(self, path, address):
#         # int JLINK_DownloadFile (const char* sFileName, U32 Addr);
#         path_str = ctypes.create_string_buffer(path)
#         retval = self.jl.JLINK_DownloadFile(path_str, ctypes.c_ulong(address))
        
        


def _locate_library(libname, paths=None, loader=None):
    if paths is None:
        paths = sys.path

    if loader is None: 
        loader = ctypes.cdll #windll
    for path in paths:
        if path.lower().endswith('.zip'):
            path = os.path.dirname(path)
        lib_path = os.path.normpath(os.path.join(path, libname))
        if os.path.exists(lib_path):
            return loader.LoadLibrary(lib_path), lib_path

    raise IOError(f'{libname} not found')



def _get_jlink_dll(library_paths):
    if library_paths is None:
        library_paths = []
    elif not isinstance(library_paths, list):
        library_paths = [library_paths]

    curos = _get_current_os()


    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Always start with the script path
    library_paths.append(script_dir)

    # Next search the MLTK commander directory
    try:
        cmder_path = download_commander()
        cmder_dir = os.path.dirname(cmder_path)
        for root, _, _ in os.walk(cmder_dir):
            library_paths.append(root)
    except:
        pass
    
    library_paths += sys.path[:]   #copy sys.path list

    # if environment variable is set, insert this path first
    try:
        library_paths.insert(0, os.environ['JLINK_PATH'])
    except KeyError:
        try:
            library_paths.extend(os.environ['PATH'].split(os.pathsep))
        except KeyError:
            pass
        
    if curos == 'win32':
        if os.path.exists('C:/Program Files/SEGGER'):
            for directory in os.listdir('C:/Program Files/SEGGER'):
                library_paths.append('C:\\Program Files\\SEGGER\\' + directory)
        elif os.path.exists('C:/Program Files (x86)/SEGGER'):
            for directory in os.listdir('C:/Program Files (x86)/SEGGER'):
                library_paths.append('C:\\Program Files (x86)\\SEGGER\\' + directory)
                
    if curos == 'win32':
        jlink, backend_info = _locate_library('jlinkarm.dll', library_paths)
   
    elif curos == 'win64':
        jlink, backend_info = _locate_library('JLink_x64.dll', library_paths)
    
    elif curos == 'linux64':
        jlink, backend_info = _locate_library('libjlinkarm.so', library_paths, ctypes.cdll)

    elif curos == 'osx':
        jlink, backend_info = _locate_library('libjlinkarm.dylib', library_paths, ctypes.cdll)
    

    return jlink, backend_info


def _get_current_os():
    if sys.platform == "linux" or sys.platform == "linux2":
        return 'linux64'
    
    elif sys.platform == "darwin":
        return 'osx'
    
    elif sys.platform == "win32":
        if sys.maxsize < 2**32:
            return 'win32'
        else:
            return 'win64'

    else:
        return 'unknown'
