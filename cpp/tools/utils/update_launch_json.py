import os 
import sys 
import argparse
import json

from mltk.utils.system import is_windows
from mltk.utils.python import load_json_safe
from mltk.utils.commander import (
    download_commander, 
    get_commander_settings, 
    get_device_from_platform
)
from mltk.utils.path import (get_user_setting, fullpath)


def main():
    parser = argparse.ArgumentParser(description='Utility to update .vscode/launch.json with a embedded debug configuration')
    parser.add_argument('--name', help='The executable name')
    parser.add_argument('--path', help='Path to the executable')
    parser.add_argument('--toolchain', help='Path to build toolchains binary directory')
    parser.add_argument('--platform', help='Name of build platform')
    parser.add_argument('--workspace', help='Path to workspace directory')
    parser.add_argument('--jlink_device', help='Values used for the JLink GDB server')
    
    args = parser.parse_args()
    vscode_dir = f'{args.workspace}/.vscode'.replace('\\', '/')
    launch_path = f'{vscode_dir}/launch.json'

    
    if os.path.exists(launch_path):
        try:
            launch_obj = load_json_safe(launch_path)
        except Exception as e:
            sys.stdout.write(f'Failed to load: {launch_path}, err: {e}')
            sys.exit(0)
    else:
        os.makedirs(vscode_dir, exist_ok=True)
        launch_obj = dict(
            version='0.2.0',
            configurations = []
        )

    if args.platform in ('windows', 'linux'):
        config_name = f'Debug {args.platform}: {args.name}'
        exe_path = args.path.replace('\\', '/')
        if os.name == 'nt':
            exe_path += '.exe'

        new_config = dict(
            name=config_name,
            stopAtEntry = True, 
            type='cppdbg',
            request='launch',
            cwd = args.workspace, 
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

    else:
        if args.jlink_device:
            device = args.jlink_device
        else:
            device = get_device_from_platform(args.platform)

        # This works around an issue in JLink not suppporting
        # the EFR32MG24BxxxF1536, however using EFR32MG24AxxxF1536 works fine
        # (even if the board is a EFR32MG24BxxxF1536)
        if device.startswith('EFR32MG24B'):
            device = 'EFR32MG24AxxxF1536'

        commander_settings = get_commander_settings()

        commander_dir = os.path.dirname(download_commander()).replace('\\', '/')

        if commander_settings['debug_server']:
            segger_server_path = commander_settings['debug_server']
        elif is_windows():
            segger_server_path = 'C:/Program Files/SEGGER/JLink/JLinkGDBServerCL.exe'
        else:
            segger_server_path = 'JLinkGDBServerCL'

        config_name = f'Debug {args.platform}: {args.name}'
        new_config = dict(
            name=config_name,
            runToEntryPoint = 'main', 
            type='cortex-debug',
            request='launch',
            servertype = 'jlink', 
            interface = 'swd',
            device = device,
            cwd = args.workspace, 
            armToolchainPath = args.toolchain.replace('\\', '/'),
            serverpath = segger_server_path.replace('\\', '/'),
            serverArgs = ['-JLinkDevicesXMLPath', commander_dir],
            executable = args.path.replace('\\', '/'),
            preRestartCommands = [
                'enable breakpoint',
                'monitor reset'
            ]
        )
        if commander_settings['ip_address']:
            new_config['ipAddress'] = commander_settings['ip_address']
        if commander_settings['serial_number']:
            new_config['serialNumber'] = commander_settings['serial_number']

        svd_path = get_user_setting(f'{args.platform}_svd_path')
        if svd_path:
            new_config['svdFile'] = fullpath(svd_path)

    found = False
    configurations = launch_obj['configurations']
    for i, cfg in enumerate(configurations):
        if cfg['name'] == config_name:
            if 'program' in configurations[i]:
                configurations[i]['program'] = new_config['program']
            elif 'executable' in configurations[i]:
                configurations[i]['executable'] = new_config['executable']
            else:
                configurations[i].update(new_config)
            found = True
            break

    if not found:
        configurations.append(new_config)

    try:
        cached_launch = None
        if os.path.exists(launch_path):
            with open(launch_path, 'r') as fp:
                cached_launch = fp.read()
            
        with open(launch_path, 'w') as fp:
            json.dump(launch_obj, fp, indent=2)

    except Exception as e:
        if cached_launch:
            with open(launch_path, 'w') as fp:
                fp.write(cached_launch)
        sys.stdout.write(f'Failed to update: {launch_path}, err: {e}')


if __name__ == '__main__':
    main()