"""Utility to program an application binary to an Silab's MCU's flash memory"""
import logging
import sys
import os
import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, Union, List, Callable

from mltk.utils.shell_cmd import run_shell_cmd
from mltk.utils.path import (create_tempdir, get_user_setting)
from mltk.utils.python import DefaultDict



if not __package__:
    CURDIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(CURDIR))
    __package__ = os.path.basename(CURDIR)# pylint: disable=redefined-builtin


from .download import download_commander


DEVICE_MAPPING = {
    'brd4166': ['EFR32MG12P332F1024GL125'],
    'brd4186': ['EFR32MG24AxxxF1536', 'EFR32MG24B210F1536IM48'],
    'brd2601': ['EFR32MG24BxxxF1536', 'EFR32MG24B110F1536GM48', 'EFR32MG24B310F1536IM48'],
    'brd2204': ['EFM32GG11B820F2048GL192']
}


@dataclass
class DeviceInfo:
    part_number:str
    die_revision:str
    production_ver:str
    flash_size:int
    sram_size:int
    unique_id:str

    @property
    def part_number_tokens(self) -> Tuple[str]:
        """Tokenize part number

        e.g. EFR32MG24B110F1536GM48
        token 0: EFR32
        token 1: MG
        token 2: 24
        token 3: B
        token 4: 110
        token 5: 1536
        token 6: GM48
        """
        match = re.match(r'.*(EFM32|EFR32)([A-Z]{2})(\d{2})([A-Z])([0-9]{3})F(\d+)([A-Z0-9]+).*', self.part_number, flags=re.IGNORECASE)
        if not match:
            raise RuntimeError(f'Invalid part number, {self.part_number}')
        return match.group(1, 2, 3, 4, 5, 6, 7)

    @property
    def flash_base_address(self) -> int:
        toks = self.part_number_tokens
        if int(toks[2]) >= 23: # Parts numbers >= 23 have a flash base address at 0x08000000
            return 0x08000000
        # Lower parts have have flash base address at 0x0
        else:
            return 0x00000000



def issue_command(
    *args,
    platform:str=None,
    outfile=None,
    line_processor: Callable[[str],str]=None,
    device:str=None,
    logger:logging.Logger=None
    ) -> str:
    """Issue a Commander command

    This has similar functionality to the command-line, e.g.:
    issue_command('--help')         -> commander.exe --help
    issue_command('device', 'info') -> commander.exe device info

    Args:
        args: Arguments to pass to Commander executable
        platform: Build platform
        outfile: File-like object used to dump Commander output
        line_processor: Callback to be invoked for each line returned by commander executable
        device: --device argument, this overrides the platform option

    Returns:
        str: Commander response string

    Raises:
        RuntimeError: Raise if command failed
    """

    cmd = _update_commander_args(*args, platform=platform, device=device)
    cmd_str = ' '.join(cmd)

    if logger is not None:
        logger.debug(cmd_str)

    retcode, retmsg = run_shell_cmd(
        cmd,
        outfile=outfile,
        line_processor=line_processor
    )
    if retcode != 0 or ('--help' not in cmd and 'error' in retmsg.lower()):
        if 'ERROR: Timeout while waiting' in retmsg and 'WARNING: DCI communication failed' in retmsg:
            pass
        elif 'more than one debugger' in retmsg:
            raise RuntimeError(
                'More than one development board is currently connected.\n' \
                'Either disconnect the other boards OR\n' \
                'update the file ~/.mltk/user_settings.yaml with your board\'s serial number and/or IP address. e.g.:\n\n' \
                'commander:\n' \
                '  ip_address: 192.168.1.3\n' \
                '  serial_number: 440139581\n\n' \
                f'Additional error details:\n{retmsg}'
                )
        else:
            if 'Cannot connect to J-Link.' in retmsg:
                retmsg += '\nThings to try:\n'
                retmsg += '- 1. Ensure the development board is properly connected and enumerated\n'
                retmsg += '- 2. Install the Segger J-Link drivers: https://www.segger.com/downloads/jlink\n'
            raise RuntimeError(f'{cmd_str}\nretcode={retcode}\n{retmsg}')

    if logger is not None:
        s = ''.join([f'{x.strip()}\n' for x in retmsg.splitlines(keepends=True) if len(x.strip()) > 0])
        logger.debug(s)

    return retmsg


def program_flash(
    path: Union[str,bytes],
    platform: str,
    address:int=None,
    show_progress=False,
    halt = False,
    reset = False,
    device:str = None,
    verify = True,
    logger:logging.Logger=None
):
    """Program flash memory of embedded device

    Args:
        path: Path to .bin/.s37 file or binary data to program to device
        platform: Name of embedded platform
        halt: If true, halt the device after programming, otherwise resume execution
        address: Device address to program, required if providing a .bin or binary data
        show_progress: Show progress while programming
        reset: Invoke software reset after programming
        device: --device argument, this overrides the platform option
        verify: Verify the flash after programming

    """

    cmd = ['flash', '--force']
    if not verify:
        cmd.append('--noverify')
    tmp_path = None
    if isinstance(path, (bytes,bytearray)):
        cmd.append('--binary')
        tmp_path = create_tempdir('tmp') + '/temp_data.bin'
        with open(tmp_path, 'wb') as f:
            f.write(path)
        path = tmp_path

    elif path.endswith('.bin'):
        cmd.append('--binary')

    if '--binary' in cmd and address is None:
        device_info = retrieve_device_info()
        address = device_info.flash_base_address

    if halt:
        cmd.append('--halt')
    if address is not None:
        cmd.extend(['--address', f'{address:08X}'])

    cmd.append(path)

    try:
        if show_progress:
            with _ProgressBar(unit = 'B', unit_scale = True, unit_divisor = 1024, miniters = 1, desc = 'Initializing') as pb:
                issue_command(
                    *cmd,
                    line_processor=pb.line_processor,
                    platform=platform,
                    device=device,
                    logger=logger
                )
        else:
            issue_command(
                *cmd,
                platform=platform,
                device=device,
                logger=logger
            )

    finally:
        if tmp_path:
            os.remove(tmp_path)

    if reset:
        reset_device(platform, device=device, logger=logger)


def reset_device(
    platform: str=None,
    device:str=None,
    logger:logging.Logger=None
):
    """Invoke software reset on device

    Args:
        platform: Name of embeddded platform
    """
    if platform is None and device is None:
        platform = query_platform()
    if logger is not None:
        logger.debug('Resetting device')
    issue_command(
        'device', 'reset',
        platform=platform,
        device=device,
        logger=logger
    )


def masserse_device(platform: str=None, device:str=None):
    """Mass erase the device's flash

    Args:
        platform: Name of embeddded platform
    """
    if platform is None and device is None:
        platform = query_platform()
    issue_command('device', 'masserase', platform=platform, device=device)


def erase_last_flash_address(device:str=None):
    """Erase the last 32-bit word of the flash memory

    This is useful to clear a programmed model
    """
    device_info = retrieve_device_info()
    platform = query_platform(device_info=device_info)
    device = device or device_info.part_number_tokens[0].lower()

    flash_end_addr = device_info.flash_base_address + (device_info.flash_size-4)
    patch_arg = f'0x{flash_end_addr:08X}:0xFFFFFFFF:4'

    issue_command('flash', '--patch', patch_arg, platform=platform, device=device)


def retrieve_device_info() -> DeviceInfo:
    """Return queried information about the device"""
    try:
        res = issue_command('device', 'info', '-d', 'efr32')
    except RuntimeError:
        res = issue_command('device', 'info', '-d', 'efm32')

    lines = res.splitlines()
    part_number = re.match(r'Part Number\s+:\s(\w+)', lines[0])
    if not part_number: # This first line could contain something else
        lines = lines[1:]  # So shift and try again
        part_number = re.match(r'Part Number\s+:\s(\w+)', lines[0])
    die_revision = re.match(r'Die Revision\s+:\s(\w+)', lines[1])
    production_ver = re.match(r'Production Ver\s+:\s(\d+)', lines[2])
    flash_size = re.match(r'Flash Size\s+:\s(\d+)\skB', lines[3])
    sram_size = re.match(r'SRAM Size\s+:\s(\d+)\skB', lines[4])
    unique_id = re.match(r'Unique ID\s+:\s(\w+)', lines[5])

    if not (part_number and die_revision and production_ver and flash_size and sram_size and unique_id):
        raise RuntimeError(f'Failed to parse commander response:\n{res}')

    info = DeviceInfo(
        part_number = part_number.group(1),
        die_revision = die_revision.group(1),
        production_ver = int(production_ver.group(1)),
        flash_size = int(flash_size.group(1))*1024,
        sram_size = int(sram_size.group(1))*1024,
        unique_id = unique_id.group(1)
    )

    return info


def query_platform(device_info:DeviceInfo = None) -> str:
    """Return the platform name of the currently connected device

    Raises:
        RuntimeError: If connected device is not supported
    """
    device_info = device_info or retrieve_device_info()

    # EFR32MG24B110F1536GM48
    # group 0: EFR32
    # group 1: MG
    # group 2: 24
    # group 3: B
    # group 4: 110
    # group 5: 1536
    # group 6: GM48
    toks = device_info.part_number_tokens

    for platform, device_codes in DEVICE_MAPPING.items():
        for device_code in device_codes:
            # For right now, just match on: EFM32xx11A / EFR32xx24B
            # This will have to get more complicated as more platforms are added
            if device_code[:5] == toks[0]:
                if device_code[7:9] == toks[2] and device_code[9] == toks[3] and device_code[10:13] == toks[4]:
                    return platform

    supported_devices = ', '.join(map(lambda x: f'{x[0]} ({x[1]})', DEVICE_MAPPING.items()))
    raise RuntimeError(f'Device not supported: {device_info.part_number}, supported devices are: {supported_devices}')



def get_device_from_platform(platform: str) -> str:
    """Given a platform name return the corresponding MCU code"""
    platform = platform.lower()
    if platform not in DEVICE_MAPPING:
        raise Exception(f'Unknown platform: {platform}, supported platforms are: {", ".join(DEVICE_MAPPING.keys())}')
    return DEVICE_MAPPING[platform][0]


def get_commander_settings() -> defaultdict:
    """Return the commander settings found in ~/.mltk/user_settings.yaml or None if not found"""
    return DefaultDict(get_user_setting('commander'))




def _update_commander_args(*args, platform:str=None, device:str=None) -> List[str]:
    """Populate commander.exe arguments from the ~/.mltk/user_settings.yaml"""
    commander_path = download_commander()
    commander_settings = get_commander_settings()

    cmd = list(args)
    cmd.insert(0, commander_path)

    file_path = None
    base_command = ''
    if len(cmd) > 1:
        base_command = cmd[1]

    if (base_command == 'flash' and '--patch' not in cmd) or (base_command == 'extflash' and cmd[2] == 'write'):
        file_path = cmd.pop()

    if base_command in ('device', 'flash', 'extflash', 'adapter', 'verify'):
        if  not ('--device' in cmd or '-d' in cmd):
            if device is not None:
                cmd.extend(['--device', device])
            elif commander_settings['device']:
                cmd.extend(['--device', commander_settings['device']])
            elif platform:
                device = get_device_from_platform(platform)
                cmd.extend(['--device', device])

        if '--ip' not in cmd:
            ip_address = commander_settings['ip_address']
            if ip_address:
                cmd.extend(['--ip', ip_address])
        if not ('--serialno' in cmd or '-s' in cmd):
            serial_number = commander_settings['serial_number']
            if serial_number:
                cmd.extend(['--serialno', serial_number])

    if file_path:
        cmd.append(file_path)

    return cmd


try:
    from tqdm import tqdm


    class _ProgressBar(tqdm):
        def __init__(self, *args, **kwargs):
            tqdm.__init__(
                self,
                *args,
                file=sys.stdout,
                bar_format='{desc}{percentage:3.0f}% | {bar} | {rate_fmt}{postfix}',
                **kwargs
            )
            self.total = 0
            self._compared = 0
            self._erased = 0
            self._downloaded = 0
            self._printed_progress = False
            self._start_re = re.compile(r'^Writing (\d+) bytes starting at address')
            self._compare_re = re.compile(r'^Comparing range 0x([0-9A-F]+) - 0x([0-9A-F]+)')
            self._erasing_re = re.compile(r'^Erasing range 0x([0-9A-F]+) - 0x([0-9A-F]+) ')
            self._programming_re = re.compile(r'^Programming range 0x([0-9A-F]+) - 0x([0-9A-F]+)')


        def line_processor(self, line):
            line_stripped = line.strip()

            if self.total == 0:
                p = self._start_re.match(line_stripped)
                if p:
                    self.total = int(p.group(1))
                return line

            p = self._compare_re.match(line_stripped)
            if p:
                self._compared += int(p.group(2), base=16) - int(p.group(1), base=16)
                self._compared = min(self._compared, self.total)
                self.set_description('Comparing', refresh=False)
                self.update(self._compared - self.n)
                return None

            p = self._erasing_re.match(line_stripped)
            if p:
                if self._erased == 0 and self.n != 0:
                    self.reset()
                self._erased += int(p.group(2), base=16) - int(p.group(1), base=16)
                self.set_description('Erasing', refresh=False)
                self.update(self._erased - self.n)
                return None

            p = self._programming_re.match(line_stripped)
            if p:
                if self._downloaded == 0 and self.n != 0:
                    self.reset()
                self._downloaded += int(p.group(2), base=16) - int(p.group(1), base=16)
                self.set_description('Downloading', refresh=False)
                self.update(self._downloaded - self.n)
                return None

            if line_stripped == 'DONE' and self._downloaded > 0:
                self.update(self.total - self.n)

            if self._printed_progress:
                self._printed_progress = False
                line ='\n' + line

            return line
except:
    pass


def main():
    parser = argparse.ArgumentParser(description='Utility to program an application binary to an Silab\'s MCUs flash memory')
    parser.add_argument('--path', default=None, help='Path to executable')
    parser.add_argument('--platform', required=True, help='MLTK platform name')
    parser.add_argument('--halt', action='store_true', default=False, help='Halt MCU after program, otherwise reset')
    parser.add_argument('--reset', action='store_true', default=False, help='Reset the device')
    parser.add_argument('--device', default=None, help='--device argument to pass to Commander')
    parser.add_argument('--masserase', action='store_true', default=False, help='Mass erase the device flash')
    parser.add_argument('--clear_end_of_flash', action='store_true', default=False, help='Erase the very last 32-bit word of the flash')

    args = parser.parse_args()

    if args.masserase:
        try:
            masserse_device(
                platform=args.platform,
                device=args.device
            )
        except:
            # Just ignore this error if it fails
            pass

    if args.clear_end_of_flash:
        try:
            erase_last_flash_address(
                device=args.device
            )
        except:
            # Just ignore this error if it fails
            pass

    if args.path:
        program_flash(
            path=args.path,
            platform=args.platform,
            halt=args.halt,
            show_progress=True,
            device=args.device
        )

    if args.reset:
        reset_device(
            platform=args.platform,
            device=args.device
        )


if __name__ == '__main__':
    main()