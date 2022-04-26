import struct
import numpy as np

from cv2 import cv2

from mltk import cli
from mltk.utils.jlink_stream import (JlinkStream, JlinkStreamOptions)
from mltk.utils.path import create_user_dir
from mltk.utils.python import install_pip_package


def main():
    logger = cli.get_logger()

    install_pip_package('opencv-python', 'cv2', logger=logger)

    dump_dir = create_user_dir('fingerprint_authenticator/dumps')

    opts = JlinkStreamOptions()
    opts.polling_period=0.100

    logger.info('Connecting to dev board via JLink ...')
    with JlinkStream(options=opts) as jlink:
        logger.info('Opening fingerprint image data stream')
        #raw_stream = jlink.open('raw')
        preprocessed_stream = jlink.open('preprocessed')

        logger.info(f'Dumping fingerprints to {dump_dir}')
        img_counter = 0
        while True:
            logger.info('Waiting for next fingerprint ...')
            # fp_img_buffer = raw_stream.read_all(192*192, timeout=-1)
            # fp_img = np.frombuffer(fp_img_buffer, dtype=np.uint8)
            # fp_img = np.reshape(fp_img, (192, 192))

            # fp_img_path = f'{dump_dir}/raw-{img_counter}.jpg'
            # logger.info(f'Saving {fp_img_path}')
            # cv2.imwrite(fp_img_path, fp_img)

            header_bytes = preprocessed_stream.read_all(4, timeout=-1)
            if header_bytes is not None:
                width, height = struct.unpack('<HH', header_bytes)
                fp_img_buffer = preprocessed_stream.read_all(width*height, timeout=-1)
                fp_img = np.frombuffer(fp_img_buffer, dtype=np.uint8)
                fp_img = np.reshape(fp_img, (height, width))

                fp_img_path = f'{dump_dir}/processed-{img_counter}.jpg'
                logger.info(f'Saving {fp_img_path}')
                cv2.imwrite(fp_img_path, fp_img)
                img_counter += 1


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass