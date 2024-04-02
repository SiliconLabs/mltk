from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Status(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    Timeout: _ClassVar[Status]
    Running: _ClassVar[Status]
    Complete: _ClassVar[Status]
    Error: _ClassVar[Status]
    DebugLog: _ClassVar[Status]
    Log: _ClassVar[Status]
    SerialOut: _ClassVar[Status]
Timeout: Status
Running: Status
Complete: Status
Error: Status
DebugLog: Status
Log: Status
SerialOut: Status

class Request(_message.Message):
    __slots__ = ("image_data", "image_path", "platform", "masserase", "device", "serial_number", "ip_address", "setup_script_data", "setup_script_args", "program_script_data", "program_script_args", "reset_script_data", "reset_script_args", "port", "baud", "timeout", "start_msg", "complete_msg", "retries", "lock_timeout")
    IMAGE_DATA_FIELD_NUMBER: _ClassVar[int]
    IMAGE_PATH_FIELD_NUMBER: _ClassVar[int]
    PLATFORM_FIELD_NUMBER: _ClassVar[int]
    MASSERASE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    SERIAL_NUMBER_FIELD_NUMBER: _ClassVar[int]
    IP_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    SETUP_SCRIPT_DATA_FIELD_NUMBER: _ClassVar[int]
    SETUP_SCRIPT_ARGS_FIELD_NUMBER: _ClassVar[int]
    PROGRAM_SCRIPT_DATA_FIELD_NUMBER: _ClassVar[int]
    PROGRAM_SCRIPT_ARGS_FIELD_NUMBER: _ClassVar[int]
    RESET_SCRIPT_DATA_FIELD_NUMBER: _ClassVar[int]
    RESET_SCRIPT_ARGS_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    BAUD_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    START_MSG_FIELD_NUMBER: _ClassVar[int]
    COMPLETE_MSG_FIELD_NUMBER: _ClassVar[int]
    RETRIES_FIELD_NUMBER: _ClassVar[int]
    LOCK_TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    image_data: bytes
    image_path: str
    platform: str
    masserase: bool
    device: str
    serial_number: str
    ip_address: str
    setup_script_data: bytes
    setup_script_args: str
    program_script_data: bytes
    program_script_args: str
    reset_script_data: bytes
    reset_script_args: str
    port: str
    baud: int
    timeout: float
    start_msg: str
    complete_msg: str
    retries: int
    lock_timeout: float
    def __init__(self, image_data: _Optional[bytes] = ..., image_path: _Optional[str] = ..., platform: _Optional[str] = ..., masserase: bool = ..., device: _Optional[str] = ..., serial_number: _Optional[str] = ..., ip_address: _Optional[str] = ..., setup_script_data: _Optional[bytes] = ..., setup_script_args: _Optional[str] = ..., program_script_data: _Optional[bytes] = ..., program_script_args: _Optional[str] = ..., reset_script_data: _Optional[bytes] = ..., reset_script_args: _Optional[str] = ..., port: _Optional[str] = ..., baud: _Optional[int] = ..., timeout: _Optional[float] = ..., start_msg: _Optional[str] = ..., complete_msg: _Optional[str] = ..., retries: _Optional[int] = ..., lock_timeout: _Optional[float] = ...) -> None: ...

class Response(_message.Message):
    __slots__ = ("status", "message")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    status: Status
    message: str
    def __init__(self, status: _Optional[_Union[Status, str]] = ..., message: _Optional[str] = ...) -> None: ...
