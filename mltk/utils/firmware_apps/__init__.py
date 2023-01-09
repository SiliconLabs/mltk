import logging
import os
import yaml
import bincopy
import struct

from mltk.core.tflite_model import TfliteModel
from mltk.cli import  is_command_active
from mltk.utils import commander
from mltk.utils.string_formatting import iso_time_str
from mltk.utils.path import create_tempdir
from mltk.utils.archive_downloader import download_verify_extract
from mltk.utils.commander import DeviceInfo

def add_image(
    name:str,
    platform:str,
    accelerator:str,
    url:str,
    sha1_hash:str,
    logger:logging.Logger
):
    """Add a firmware image to the download_urls.yaml file

    Args:
        name: Name of image
        platform: Embedded platform name
        accelerator: Name of hardware accelerator built into image
        url: Download URL
        sha1_hash: SHA1 hash of downloaded file
        logger: Logger
    """
    curdir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
    download_urls_file = f'{curdir}/download_urls.yaml'

    try:
        with open(download_urls_file, 'r') as fp:
            download_urls = yaml.load(fp, Loader=yaml.SafeLoader)
            if not download_urls:
                raise Exception()
    except:
        download_urls = {}

    key = get_url_key(
        name=name,
        platform=platform,
        accelerator=accelerator
    )

    download_urls[key] = dict(
        url=url,
        sha1=sha1_hash,
        date=iso_time_str()
    )

    with open(download_urls_file, 'w') as fp:
        yaml.dump(download_urls, fp, Dumper=yaml.SafeDumper)

    logger.info(f'Updated {download_urls_file}')


def get_image(
    name:str,
    platform: str,
    accelerator: str,
    logger:logging.Logger
) -> str:
    """Return the path to a firmware image"""

    curdir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
    download_urls_file = f'{curdir}/download_urls.yaml'
    key = get_url_key(
        name=name,
        platform=platform,
        accelerator=accelerator
    )

    try:
        with open(download_urls_file, 'r') as fp:
            download_urls = yaml.load(fp, Loader=yaml.SafeLoader)
    except Exception as e:
        # pylint: disable=raise-missing-from
        raise Exception(f'Failed to load: {download_urls_file}, err: {e}')

    if not download_urls or key not in download_urls:
        if accelerator:
            raise RuntimeError(f'The hardware platform: {platform} does not support the accelerator: {accelerator}')

        raise RuntimeError(f'The hardware platform: {platform} does not support the application: {name}')

    image_info = download_urls[key]
    download_dir = download_verify_extract(
        url=image_info['url'],
        file_hash=image_info['sha1'],
        file_hash_algorithm='sha1',
        dest_subdir=f'firmware/{key}',
        show_progress=is_command_active(),
        logger=logger,
    )

    img_ext = get_image_extension(platform=platform)
    return f'{download_dir}/{key}{img_ext}'


def program_image_with_model(
    name:str,
    accelerator:str,
    tflite_model:TfliteModel,
    logger: logging.Logger,
    platform:str=None,
    halt:bool = False,
    firmware_image_path:str = None,
    model_offset:int=0,
    show_progress:bool=None
):
    """Program a FW image + .tflite model to an embedded device"""
    platform = platform or commander.query_platform()
    if show_progress is None:
        show_progress = is_command_active()

    if firmware_image_path is None:
        firmware_image_path = get_image(
            name=name,
            platform=platform,
            accelerator=accelerator,
            logger=logger
        )


    firmwage_image_length = -1
    if firmware_image_path:
        if firmware_image_path.lower() != 'none':
            firmwage_image_length = _get_firmware_image_size(firmware_image_path)

            # Program the generated .s37
            # and halt the CPU after programming
            logger.info(f'Programming FW image: {name} to device ...')
            commander.program_flash(
                firmware_image_path,
                platform=platform,
                show_progress=show_progress,
                halt=halt,
            )
        else:
            logger.info('Not programming firmware image to device')


    if tflite_model is not None:
        program_model(
            tflite_model=tflite_model,
            logger=logger,
            platform=platform,
            halt=halt,
            firmwage_image_length=firmwage_image_length,
            offset=model_offset
        )


def program_model(
    tflite_model:TfliteModel,
    logger: logging.Logger,
    platform:str = None,
    halt:bool = False,
    firmwage_image_length:int=-1,
    offset:int=0,
    show_progress:bool=None,
):
    """Program the .tflite model to the end of the device's flash"""
    if show_progress is None:
        show_progress = is_command_active()

    try:
        device_info = commander.retrieve_device_info()
    except:
        logger.warning('Failed to retrieve device info')

    if platform is None:
        platform = commander.query_platform(device_info=device_info)

    tmp_tflite_path = create_tempdir('temp') + '/model.bin'

    bin_data = bytearray(tflite_model.flatbuffer_data)
    tflite_length = len(bin_data)
    bin_data.extend(struct.pack('<L', tflite_length))

    with open(tmp_tflite_path, 'wb') as f:
        f.write(bin_data)

    flash_size = device_info.flash_size

    if firmwage_image_length + len(bin_data) + offset > flash_size:
        raise RuntimeError(f'Flash overflow, the .tflite model size ({len(bin_data)}) + app size ({firmwage_image_length}) exceeds the flash size ({flash_size})')


    flash_end_address = device_info.flash_base_address + flash_size
    flash_program_address = flash_end_address - len(bin_data) - offset
    logger.info(f'Programming .tflite to flash address: 0x{flash_program_address:08X}')
    commander.program_flash(
        tmp_tflite_path,
        address=flash_program_address,
        platform=platform,
        show_progress=show_progress,
        halt=halt,
        logger=logger
    )
    os.remove(tmp_tflite_path)


def get_url_key(name:str, platform:str, accelerator: str) -> str:
    name = name.lower()
    platform = platform.lower()
    accelerator = accelerator or 'none'
    accelerator = accelerator.lower()
    return f'{name}-{platform}-{accelerator}'


def get_image_extension(platform:str) -> str:
    """Return the image extension used by the given platform"""
    if platform == 'windows':
        return '.exe'
    elif platform in ('linux', 'osx'):
        return ''
    else:
        return '.s37'


def _get_firmware_image_size(path:str) -> int:
    """Return the size of the firmware image in bytes"""
    if path.endswith('.s37'):
        s37 = bincopy.BinFile()
        try:
            s37.add_srec_file(path)
        except Exception as e:
            raise Exception(f'Failed to process firmwage image, is it a valid .s37 file? Error details: {e}')

        return len(s37) * s37.word_size_bytes

    elif path.endswith('.bin'):
        return os.path.getsize(path)

    raise RuntimeError('Unsupported file type')