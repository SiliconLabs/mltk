
import os
import logging
import io
import struct
import argparse
import yaml
from mltk.utils.uart_stream import UartStream
from mltk.utils.python import install_pip_package


# This is a work-around for the AlexaClient Python package that uses an older Python version
import collections.abc
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping


# Install the 'alexa_client' Python package:
# https://pypi.org/project/alexa-client
install_pip_package('alexa_client')
from alexa_client.refreshtoken import serve as avs_refreshtoken
from alexa_client import AlexaClient
from alexa_client.alexa_client import constants as avs_constants


# Configure the logger
logging.basicConfig(level='INFO',  handlers=[logging.StreamHandler()])
logging.getLogger('hpack').setLevel('ERROR')
logging.getLogger('hyper').setLevel('ERROR')
logger = logging.getLogger('AlexaDemo')



def main(
    client_id:str,
    secret:str,
    refresh_token:str,
    baud:int
):
    """This is the main entry point of the script"""

    logger.info('Alexa demo starting ...')
    alexa_client = AlexaClient(
        client_id=client_id,
        secret=secret,
        refresh_token=refresh_token,
        base_url=avs_constants.BASE_URL_NORTH_AMERICA
    )
    # Connect to the AVS cloud
    alexa_client.connect()
    try:
        application_loop(alexa_client, baud=baud)
    except KeyboardInterrupt:
        pass
    finally:
        alexa_client.ping_manager.cancel()
        alexa_client.connection_manager.connection.close()


def application_loop(
    alexa_client: AlexaClient,
    baud:int
):
    """This is the main application loop

    It retrieves commands from the development board,
    forwards them to the AVS cloud, and forwards the AVS responses
    to the development board
    """
    with DevBoard(baud=baud) as board:
        dialog_request_id = None

        while True:
            logger.info('Waiting for an "Alexa" command. (speak into the dev board microphone)')
            alexa_command = board.wait_for_command()

            logger.info('Sending "Alexa" command to AVS cloud')
            directives = alexa_client.send_audio_file(
                alexa_command,
                dialog_request_id=dialog_request_id,
                audio_format=avs_constants.OPUS,
                distance_profile=avs_constants.CLOSE_TALK
            )
            if not directives:
                logger.warning('Failed to receive response from AVS')
                continue

            dialog_request_id = None

            for directive in directives:
                if directive.name == 'ExpectSpeech':
                    logger.info('Expecting a response from user ...')
                    dialog_request_id = directive.dialog_request_id
                    board.start_microphone(at_end_of_speaker_audio=True)

                elif directive.name in ['Speak', 'Play']:
                    logger.info(f'Received response from AVS cloud: {len(directive.audio_attachment)} bytes')
                    board.play_audio(directive.audio_attachment)





class DevBoard:
    """This is a helper class to communicate with the dev board's UART"""

    # These commands are also defined in audio_io.cc
    CMD_START_MICROPHONE = 0
    CMD_STOP_MICROPHONE = 1
    CMD_START_MICROPHONE_AT_END_OF_SPEAKER_AUDIO = 2
    CMD_PLAY_AUDIO = 3

    def __init__(
        self,
        baud: 115200
    ):
        self._uart = UartStream(baud=baud)

    def open(self):
        self._uart.open()


    def close(self):
        self._uart.close()

    def wait_for_command(self) -> io.BytesIO:
        """Wait and retrieve a command spoken to the dev board's microphone
        and return the Opus-encoded audio as a binary file object
        """
        data_buffer = io.BytesIO()

        while True:
            cmd = self._uart.read_command()

            if cmd.code == self.CMD_STOP_MICROPHONE:
                logger.info(f'Command received: {data_buffer.tell()} bytes')
                self._uart.flush_input()
                data_buffer.seek(0)
                return data_buffer

            data = self._uart.read()
            if not data:
                self._uart.wait(0.100)
                continue

            if data_buffer.getbuffer().nbytes == 0:
                logger.info('Receiving command ...')

            data_buffer.write(data)


    def play_audio(self, mp3_data:bytes):
        """Send the MP3-encoded audio to the development board"""
        self._uart.write_command(self.CMD_PLAY_AUDIO, struct.pack('<l', len(mp3_data)))
        self._uart.write_all(mp3_data)

    def start_microphone(self, at_end_of_speaker_audio=False):
        """Command the dev board to start streaming microphone audio"""
        self._uart.write_command(
            self.CMD_START_MICROPHONE_AT_END_OF_SPEAKER_AUDIO if at_end_of_speaker_audio
            else self.CMD_START_MICROPHONE
    )

    def __enter__(self):
        logger.info(f'Opening UART connection to development board (BAUD={self._uart.baud}) ...')
        self.open()
        logger.info('UART connection opened')
        return self

    def __exit__(self, dtype, value, tb):
        self.close()



def generate_credentials(credentials_path:str):
    """This is a helper function to generate the avs_credentials.yaml file"""

    import re
    import threading

    credentials = {
        'device-id': '',
        'client-id': '',
        'client-secret': '',
        'refresh-token': ''
    }

    print('Before running this demo, we need to create the file:')
    print(credentials_path)
    print('which contains your Alexa Voice credentials.\n')

    print('\n1) Go to:\n\nhttps://developer.amazon.com/settings/console/registration\n')
    print('and either create or sign into your Amazon developer account.')
    input('\nPress ENTER to continue')

    print('\n2) Next goto:\n\nhttps://developer.amazon.com/alexa/console/avs/products/new\n')
    print('and create a new "Alexa product".')
    print('Be sure to check: "Device with Alexa built-in" for the "product type"')
    print('All the other fields may be anything you like.')
    print('Be sure to note the "Product ID" you select and enter it in the following prompt:')

    while True:
        device_id = input('\nEnter your "Product ID" then press ENTER: ').strip()
        if not re.match(r'^[\da-zA-Z_]+$', device_id):
            print('  Invalid Product ID\n')
            continue
        break
    credentials['device-id'] = device_id

    print('\n3) After clicking the "Next" (or "Update") button on the webpage of the previous step,')
    print('you\'ll be instructed to select an existing or create a new "Security Profile",')
    print("Follow the webpage's instructions. (if you\'re not prompted, then go to the \"Security Profile\" tab on the left sidebar)")
    input('\nthen, press ENTER to continue')

    print('\n4) After the "Security Profile" is selected, copy the "Client ID"')
    print('and paste it into the following prompt:\n')

    while True:
        client_id = input('\nEnter your "Client ID" then press ENTER: ').strip()
        if not re.match(r'^amzn1\.application-oa2-client\.[\da-z]+$', client_id):
            print('  Invalid Client-ID\n')
            continue
        break
    credentials['client-id'] = client_id


    print('\n5) Next, copy the "Client secret"')
    print('and paste it into the following prompt:\n')

    while True:
        client_secret = input('\nEnter your "Client secret" then press ENTER: ').strip()
        if not re.match(r'^[\da-z]+$', client_secret):
            print('  Invalid Client-secret\n')
            continue
        break
    credentials['client-secret'] = client_secret


    print('\n6) At the bottom of the webpage, where it says "Allowed origins", add the entry (if necessary):')
    print('\nhttp://localhost:9000')
    input('\nPress ENTER to continue')


    print('\n7) At the bottom of the webpage, where it says "Allowed return URLs", add the entry (if necessary):')
    print('\nhttp://localhost:9000/callback/')
    input('\nPress ENTER to continue')

    print('\n8) Click the button "Update Web Settings" (if necessary)')
    input('\nPress ENTER to continue')

    print('\n9) At the bottom of the webpage, click the "I agree ..." checkbox then click the "Finish" button. (if necessary)')
    input('\nPress ENTER to continue')

    print('\n10) Next, go to:\n\nhttp://localhost:9000\n\nand follow the instructions provided by the webpage')

    server_address = ('localhost', 9000)
    callback_url = 'http://{}:{}/callback/'.format(*server_address)

    def _log_message(self, format, *args):
        return
    avs_refreshtoken.handlers.AmazonAlexaServiceLoginHandler.log_message = _log_message
    server = avs_refreshtoken.http_server.AmazonLoginHttpServer(
        server_address=server_address,
        RequestHandlerClass=avs_refreshtoken.handlers.AmazonAlexaServiceLoginHandler,
        client_id=client_id,
        client_secret=client_secret,
        device_type_id=device_id,
        callback_url=callback_url,
    )
    t = threading.Thread(
        target=server.serve_forever,
        daemon=True
    )
    t.start()
    input('\nPress ENTER to continue')

    print('\n10) At the very end, the http://localhost:9000 webpage will display a "refresh_token", copy its value into the following prompt:\n')
    while True:
        refresh_token = input('\nEnter the "refresh_token" then press ENTER: ').strip()
        if refresh_token.startswith('refresh_token: '):
            refresh_token = refresh_token.replace('refresh_token:', '').strip()
        if not re.match(r'^[a-zA-Z\d_\|\-]+$', refresh_token):
            print('  Invalid refresh-token\n')
            continue
        break
    credentials['refresh-token'] = refresh_token

    print(f'Generating {credentials_path}')
    with open(credentials_path, 'w') as config_file:
        yaml.dump(credentials, config_file, Dumper=yaml.SafeDumper)

    print('Shutting down local server ...')
    server.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Alexa Voice Services demo, see https://siliconlabs.github.io/mltk/mltk/tutorials/keyword_spotting_alexa.html')
    parser.add_argument('--setup', action='store_true', default=False, help='Setup the AVS credentials')
    parser.add_argument('--baud', default=115200, help='Specify the UART BAUD rate', type=int, metavar='<rate>')
    args = parser.parse_args()

    curdir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
    credentials_path = f'{curdir}/avs_credentials.yaml'
    if not os.path.exists(credentials_path) or args.setup:
        generate_credentials(credentials_path)

    with open(credentials_path, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)

    main(
        client_id=config['client-id'],
        secret=config['client-secret'],
        refresh_token=config['refresh-token'],
        baud=config.get('baud', args.baud)
    )