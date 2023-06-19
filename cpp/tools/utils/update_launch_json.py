import os
import sys
import argparse
import json
import traceback

from mltk.utils.system import is_windows
from mltk.utils.python import load_json_safe
from mltk.utils.commander import (
    download_commander,
    get_commander_settings,
    get_device_from_platform
)
from mltk.utils.path import (get_user_setting, fullpath)




def update_launch_json(
    name:str,
    exe_path:str,
    toolchain_dir:str,
    platform:str,
    workspace_dir:str,
    device:str=None,
    ip_address:str=None,
    serial_number:str=None,
    interface:str='swd',
    speed:str='auto',
    svd_path:str=None
):
    """
    Update the .vscode/launch.json

    Args:
        name: CMake target name
        exe_path: Path to embedded executable with symbols (.elf, .axf, etc.)
        toolchain: Path to build toolchains binary directory
        platform: Name of build platform
        workspace: VSCode workspace directory
        device: J-Link device code
    """

    vscode_dir = fullpath(f'{workspace_dir}/.vscode')
    launch_json_path = f'{vscode_dir}/launch.json'
    exe_path = fullpath(exe_path)
    if toolchain_dir:
        toolchain_dir = fullpath(toolchain_dir)
    workspace_dir = fullpath(workspace_dir)

    if os.path.exists(launch_json_path):
        try:
            launch_obj = load_json_safe(launch_json_path)
        except Exception as json_load_ex:
            raise RuntimeError(f'Failed to load: {launch_json_path}, err: {json_load_ex}')
    else:
        os.makedirs(vscode_dir, exist_ok=True)
        launch_obj = dict(
            version='0.2.0',
            configurations = []
        )

    if platform in ('windows', 'linux'):
        config_name = f'Debug {platform}: {name}'
        if os.name == 'nt' and not exe_path.endswith('.exe'):
            exe_path += '.exe'

        new_config = dict(
            name=config_name,
            stopAtEntry = True,
            type='cppdbg',
            request='launch',
            cwd = workspace_dir,
            program = exe_path,
            externalConsole = False,
            MIMode = 'gdb',
            setupCommands = [
                dict(
                    description='Enable pretty-printing for gdb',
                    text='-enable-pretty-printing'
                ),
                dict(
                    description='Break on exceptions',
                    text='catch throw'
                )
            ]
        )
        if toolchain_dir and platform in ('windows',):
            new_config['miDebuggerPath'] = f'{toolchain_dir}/gdb.exe'

    else:
        if not device or device == 'auto':
            device = get_device_from_platform(platform)

        # This works around an issue in JLink not supporting
        # the EFR32MG24BxxxF1536, however using EFR32MG24AxxxF1536 works fine
        # (even if the board is a EFR32MG24BxxxF1536)
        if device and device.startswith('EFR32MG24B'):
            device = 'EFR32MG24AxxxF1536'

        commander_settings = get_commander_settings()

        commander_dir = os.path.dirname(download_commander()).replace('\\', '/')

        if commander_settings['debug_server']:
            segger_server_path = commander_settings['debug_server']
        elif is_windows():
            segger_server_path = 'C:/Program Files/SEGGER/JLink/JLinkGDBServerCL.exe'
        else:
            segger_server_path = 'JLinkGDBServerCL'

        config_name = f'Debug {platform}: {name}'
        new_config = dict(
            name=config_name,
            runToEntryPoint = 'main',
            type='cortex-debug',
            request='launch',
            servertype = 'jlink',
            interface = interface,
            device = device,
            cwd = workspace_dir,
            armToolchainPath = toolchain_dir,
            serverpath = segger_server_path.replace('\\', '/'),
            serverArgs = ['-JLinkDevicesXMLPath', commander_dir],
            executable = exe_path,
            preRestartCommands = [
                'enable breakpoint',
                'monitor reset'
            ]
        )

        if ip_address:
            new_config['ipAddress'] = ip_address
        elif commander_settings['ip_address']:
            new_config['ipAddress'] = commander_settings['ip_address']

        if serial_number:
            new_config['serialNumber'] = serial_number
        elif commander_settings['serial_number']:
            new_config['serialNumber'] = commander_settings['serial_number']


        if speed:
            new_config['serverArgs'].extend(['-speed', speed])

        svd_path = get_user_setting(f'{platform}_svd_path', default=svd_path)
        if svd_path:
            new_config['svdFile'] = fullpath(svd_path)

    already_exists = False
    configurations = launch_obj['configurations']
    for i, cfg in enumerate(configurations):
        if cfg['name'] == config_name:
            existing_config = configurations[i]
            update_keys = ('program', 'executable', 'serialNumber', 'ipAddress', 'armToolchainPath')
            for key in update_keys:
                if key in new_config:
                    existing_config[key] = new_config[key]
                    already_exists = True
            break

    if not already_exists:
        configurations.append(new_config)

    cached_launch_json = None
    try:
        if os.path.exists(launch_json_path):
            with open(launch_json_path, 'r') as fp:
                cached_launch_json = fp.read()

        with open(launch_json_path, 'w') as fp:
            json.dump(launch_obj, fp, indent=2)

    except Exception as e:
        if cached_launch_json:
            with open(launch_json_path, 'w') as fp:
                fp.write(cached_launch_json)
        raise RuntimeError(f'Failed to update: {launch_json_path}, err: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utility to update .vscode/launch.json with a embedded debug configuration')
    parser.add_argument('--name', help='The executable name')
    parser.add_argument('--path', help='Path to the executable')
    parser.add_argument('--toolchain', help='Path to build toolchains binary directory')
    parser.add_argument('--platform', help='Name of build platform')
    parser.add_argument('--workspace', help='Path to workspace directory')
    parser.add_argument('--device', help='Device options to pass to JLink GDB server')
    parser.add_argument('--serial_number', help=' JLink debugger serial number')
    parser.add_argument('--ip_address', help=' JLink debugger IP address')
    parser.add_argument('--interface', help=' JLink debugger interface (jtag, swd, etc)', default='swd')
    parser.add_argument('--speed', help=' JLink debugger speed', default='auto')
    parser.add_argument('--svd_path', help='File path to .svd file for given platform')


    args = parser.parse_args()

    try:
        update_launch_json(
            name=args.name,
            exe_path=args.path,
            platform=args.platform,
            workspace_dir=args.workspace,
            toolchain_dir=args.toolchain,
            device=args.device,
            serial_number=args.serial_number,
            ip_address=args.ip_address,
            interface=args.interface,
            speed=args.speed,
            svd_path=args.svd_path
        )
    except Exception as cli_ex:
        traceback.print_exc()
        sys.stdout.write(f'{cli_ex}')
        sys.exit(-1)