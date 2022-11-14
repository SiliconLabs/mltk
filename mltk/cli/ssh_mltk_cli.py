import os
from enum import Enum
import typer

from mltk import cli


@cli.root_cli.command('ssh', context_settings=dict(allow_extra_args=True, ignore_unknown_options=True))
def ssh_command(
    host: str = typer.Option(None, '-h', '--host',
        help='''\b
SSH hostname. With format: [<user name>@]<host>[:<port>][/<path>]
\b
Where:
- <user name> - Optional, user login name
- <host> - Required, SSH hostname
- <port> - Optional, SSH port, default is 22
- <path> - Optional, remote directory path
\b
Examples:
--host myserver.com
--host 192.168.1.56 
--host ubuntu@192.168.1.56 
--host ubuntu@192.168.1.56:456
--host ubuntu@192.168.1.56/workspace
\b
If omitted, then use the settings defined in ~/.mltk/user_settings.yaml
\b
NOTE: If the given hostname exists in ~/.ssh/config or the config file path defined ~/.mltk/user_settings.yaml, then the settings defined there will be used
''',
        metavar='<host>'
    ),
    identity_file: str = typer.Option(None, '-i', '--identity_file',
        help='''Path to SSH private key file in PEM format. If omitted, then use the settings defined in ~/.mltk/user_settings.yaml or ~/.ssh/config''',
        metavar='<key>'
    ),
    port: int = typer.Option(None, '-p', '--port',
        help='''SSH server listening port. If omitted, then use the --host port or settings defined in ~/.mltk/user_settings.yaml or ~/.ssh/config''',
        metavar='<port>'
    ),
    password: str = typer.Option(None, '--password',
        help='''SSH user password. If omitted, then use the settings defined in ~/.mltk/user_settings.yaml''',
        metavar='<password>'
    ),
    resume: bool = typer.Option(False,
        help='''Resume a previously executed command. This is useful if the SSH connection prematurely disconnects or using the --no-wait option'''
    ),
    wait: bool = typer.Option(True,
        help='''Wait for the command to complete on the remote server. If using --no-wait, then --resume can later be used to retrieve the command results'''
    ),
    force: bool = typer.Option(False,
        help='''Only one command can be active on the remote server. Use this to abort/discard a previously executed command'''
    ),
    clean: bool = typer.Option(False,
        help='''If running the "train" command, then clean the local AND remote model log directories before invoking command.  This will also clean the existing model archive on the local machine'''
    ),
    verbose: bool = typer.Option(False, '-v', '--verbose',
        help='''Enable verbose logging'''
    ),
    mltk_command: str = typer.Argument(..., 
        help='''\b
MLTK command to run on remote SSH server
The currently supported commands are: train
''',
        metavar='<command>'
    ),
    mltk_cmd_args: typer.Context = typer.Option(None),
):
    """Execute an MLTK command on a remote SSH server

    \b
    This allows for executing an MLTK command on a remote SSH server.
    This is useful for training a model in the "cloud" and then downloading
    the training results back to the local computer.
    \b
    The command executes the basic flow:
    \b
    1. Open SSH connection to remote server
       Using the settings specified in the --host option, in ~/.mltk/user_settings.yaml, or in ~/.ssh/config
    2. Create remote working directory
       Specified in --host option or in ~/.mltk/user_settings.yaml
    3. Create and activate an MLTK python virtual environment 
       (if not disabled in model specification or ~/.mltk/user_settings.yaml)
    4. Upload files configured in model specification and/or ~/.mltk/user_settings.yaml
    5. Export any environment variables configured in model specification and/or ~/.mltk/user_settings.yaml
    6. Execute any startup shell commands configured in model specification and/or ~/.mltk/user_settings.yaml
    7. Execute the MLTK command in a detached subprocess
       This way, the command continues to execute even if the SSH session prematurely closes
    8. If --no-wait was NOT given, then poll the remote MLTK command subprocess while dumping the remote log file to the local terminal
       Issuing CTRL+C will abort the remote command subprocess
    9. Once the MLTK command completes, download the model archive file (if available)
    10. Download any files configured in model specification and/or ~/.mltk/user_settings.yaml
    11. Download any other logs files
    12. Execute any shutdown shell commands configured in model specification and/or ~/.mltk/user_settings.yaml
    \b
    # Example Usage
    \b
    # Train model on the remote SSH server using the SSH credentials configured in ~/.mltk/user_settings.yaml.
    # After training completes, the model will be copied to the local machine
    mltk ssh train keyword_spotting_on_off_v2
    \b
    # Train model on the remote SSH server using the SSH credentials provided on the command-line.
    # After training completes, the model will be copied to the local machine
    mltk ssh --host root@ssh4.vast.ai -p 18492 -i ~/.ssh/id_vast_ai train keyword_spotting_on_off_v2
    \b
    # Start model training but do NOT wait for it to complete.
    # The model will train on the remote server
    # We can later poll the results using the --resume option
    # NOTE: In this example, the SSH settings are stored in the ~/.mltk/user_settings.yaml file
    mltk ssh train keyword_spotting_on_off_v2 --no-wait
    \b
    # Retrieve the results of a previously started command
    # This can be used if the SSH connection prematurely disconnects OR if --no-wait was previously called
    mltk ssh train keyword_spotting_on_off_v2 --resume
    \b
    # Train a model and discard a previously invoked command
    mltk ssh train audio_example1 --force
    \b
    # SSH Settings
    \b
    The various SSH settings may be configured in the model mixin: SshMixin and/or in the file ~/.mltk/user_settings.yaml and/or ~/.ssh/config.
    \b
    For more details, see:
    https://siliconlabs.github.io/mltk/docs/guides/model_training_via_ssh.html
    \b
    NOTE: Issuing Ctrl+C will cancel the command on the remote server

    """

    from mltk.utils import ssh
    from mltk.cli.utils import log_file
  

    SUPPORTED_COMMANDS = ('train', )

    logger = cli.get_logger(verbose=verbose)
    
    if mltk_command not in SUPPORTED_COMMANDS:
        cli.abort(msg=f'Unsupported MLTK command, supported commands are: {", ".join(SUPPORTED_COMMANDS)}')

    if len( mltk_cmd_args.args) == 0:
        cli.abort(msg='Must provide model argument, e.g.: mltk ssh train my_model')

    cmd  = [mltk_command] + list(mltk_cmd_args.args)

    try:
        if not verbose:
            cli.print_info(f'HINT: For verbose remote server logs, see: {log_file}')
        
        else:
            cmd.append('--verbose')

        ssh.run_mltk_command(
            ssh_host=host,
            ssh_port=port,
            ssh_key_path=identity_file,
            ssh_password=password,
            cmd=cmd,
            clean=clean,
            force=force, 
            resume_only=resume, 
            wait_for_results=wait, 
            logger=logger
        )

    except KeyboardInterrupt:
        pass

    except Exception as e:
        cli.handle_exception('Failed to run SSH command', e)
    
    finally:
        cli.print_info('\nDo not forget to shutdown your cloud instance when you\'re done! (if applicable)')
    

class KeyType(str, Enum):
    rsa = 'rsa'
    ed25519 = 'ed25519'


@cli.root_cli.command('ssh-keygen')
def ssh_gen_key_command(
    name:str = typer.Argument(...,
        help='The name of the key',
        metavar='<name>'
    ),
    key_type:KeyType = typer.Option(KeyType.ed25519, '-t', '--type', 
        case_sensitive=False,
        metavar='<type>'
    ),
    output:str = typer.Option('~/.ssh', '--output', '-o',
        help='Output directory where keypair will be generated',
        metavar='<output directory>'
    ),
    key_length:int = typer.Option(3072, '--length', '-l',
        help='Length of RSA key',
        metavar='<length>'
    )
):
    """Generate an SSH keypair
    
    \b
    This is a helper command to generate an keypair for an SSH connection.
    \b
    # Example:
    # Generate an Ed25519 keypair with name: my_server
    mltk ssh-keygen my_server
    # Generate an RSA 3072-bit keypair with name: my_server
    mltk ssh-keygen -t rsa my_server
    """
    from mltk.utils.path import fullpath

    try:
        import cryptography
    except:
        raise RuntimeError('Failed import cryptography Python package, try running: pip install cryptography OR pip install silabs-mltk[full]')

    from cryptography.hazmat.primitives import serialization as crypto_serialization
    from cryptography.hazmat.primitives.asymmetric import (rsa, ed25519)
    from cryptography.hazmat.backends import default_backend as crypto_default_backend

    if key_type == KeyType.rsa:
        key = rsa.generate_private_key(
            backend=crypto_default_backend(),
            public_exponent=65537,
            key_size=key_length
        )
    elif key_type == KeyType.ed25519:
        key = ed25519.Ed25519PrivateKey.generate()

    private_key = key.private_bytes(
        crypto_serialization.Encoding.PEM,
        crypto_serialization.PrivateFormat.OpenSSH,
        crypto_serialization.NoEncryption()
    )

    public_key = key.public_key().public_bytes(
        crypto_serialization.Encoding.OpenSSH,
        crypto_serialization.PublicFormat.OpenSSH
    ).decode('utf-8')

    os.makedirs(fullpath(output), exist_ok=True)
    private_key_path = fullpath(f'{output}/id_{name}')
    public_key_path = fullpath(f'{output}/id_{name}.pub')

    cli.print_info(f'Generating {private_key_path}')
    with open(private_key_path, 'wb') as f:
        f.write(private_key)

    cli.print_info(f'Generating {public_key_path}')
    public_key = f'{public_key} {name}'
    with open(public_key_path, 'w') as f:
        f.write(public_key)

    cli.print_info(f'Public key:\n{public_key}')

