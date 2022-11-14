# Model Training via SSH

This describes how to train a machine learning model on a remote machine via [SSH](https://en.wikipedia.org/wiki/Secure_Shell).
This is useful as it allows for developing a model on a local, resource-constrained machine and then seamlessly train 
the model on a much larger remote "cloud" server.


## Quick Reference

- Command-line: `mltk ssh --help`
- Tutorial: [Cloud Training with vast.ai](../../mltk/tutorials/cloud_training_with_vast_ai.ipynb)


## Overview

The MLTK features the command: `mltk ssh` which internally manages all of the details necessary to execute an MLTK command on a remote machine.  

The basic flow for training a model in the cloud is as as follows:  
![](../img/model_training_via_ssh.png)


1. Create a [model specification](./model_specification.md) on a local machine
2. Invoke the command: `mltk ssh train my_model`, which will:  
   a. Open a secure connection to a remote machine  
   b. Upload all necessary files to the remote machine  
3. Invoke the [train](./model_training.md) command on the remote machine (which may have a large amount of GPUs/CPUs/RAM)
4. After training completes on the remote machine, the [model archive](./model_archive.md) and any other training files are downloaded to the local machine

So basically, develop the model on the local machine, quickly train it on a cloud machine, and all training results appear on the local machine as if the model had been trained locally.


## SSH Connection

[SSH](https://en.wikipedia.org/wiki/Secure_Shell) is a standard protocol for securely connecting to remote machines. 
With it, shell commands may be issued from a local machine and executed on a remote machine.

While the details of creating an SSH connection is out-of-scope for this document, it is important to note the following:  
- The __SSH Server__ runs on the __remote machine__
- The __SSH Client__ runs on the __local machine__
- [OpenSSH](https://www.openssh.com/) is a free, open-source tool that provides both the an SSH client and server which are available for Windows, Linux, and Mac.
- The OS of the client does _not_ need to match the server, e.g. a Windows SSH client can connect to a Linux SSH server

### Installing an SSH client

While an SSH client does _not_ need to be installed on the local machine, 
it is helpful to have one to ensure the login credentials are working before using the `mltk ssh` command (which internally uses its own SSH client [python library](https://www.paramiko.org/)).

#### Windows

Refer to the following documentation for how to install the SSH client on Windows: [Get started with OpenSSH](https://docs.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse)

#### Linux

On Linux, the SSH client is likely installed by default. However, on Ubuntu-like systems, it can be installed with:

```shell
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install openssh-client
```

### Generating a Keypair

A [keypair](https://wiki.archlinux.org/title/SSH_keys) is required to securely connect to the SSH server.
The details of creating and distributing the keypair are out-of-scope for this document, however, it is important to note the following:  
- A keypair consists of one __private key__ and one __public key__
- The __private key__ resides on the __local machine__, its contents must be securely stored (i.e. do not share it with others)
- The __public key__ resides on the __remote machine__, its contents need not be secure (i.e. it can be copied & pasted anywhere)

#### Command

The MLTK features a helper command for generating an SSH keypair:

```shell
mltk ssh-keygen --help
```

Which will generate an Ed25519 keypair in the specified output directory., e.g.:

```shell
# Generate pair at: ~/.ssh/id_my_key
mltk ssh-keygen my_key
```

#### Additional Resources

Refer to the following for additional information on creating and distributing a keypair:  
- [Windows OpenSSH Key Management](https://docs.microsoft.com/en-us/windows-server/administration/openssh/openssh_keymanagement)
- [Generate SSH keys on Linux](https://linuxhint.com/generate-ssh-keys-on-linux/)


## Command sequence

When the `mltk ssh <command>` command is invoked, the following sequence is internally executed:

1. Open an SSH connection to remote server  
   Using the settings specified in the `--host` option, in `~/.mltk/user_settings.yaml`, or in `~/.ssh/config`
2. Create remote working directory  
   Specified in `--host` option or in `~/.mltk/user_settings.yaml`
3. Create and activate an MLTK python virtual environment   
   Only if not disabled in model specification or `~/.mltk/user_settings.yaml`
4. Upload files configured in model specification and/or `~/.mltk/user_settings.yaml`
5. Export any environment variables configured in model specification and/or `~/.mltk/user_settings.yaml`
6. Execute any startup shell commands configured in model specification and/or `~/.mltk/user_settings.yaml`
7. Execute the MLTK `<command>` in a detached subprocess  
   This way, the command continues to execute even if the SSH session prematurely disconnects
8. Poll the remote MLTK command subprocess while dumping the remote log file to the local terminal  
   Issuing CTRL+C will abort the remote command subprocess
   (Use the `--no-wait` option to skip this step)
9. Once the MLTK command completes, download the model archive file (if available)
10. Download any files configured in model specification and/or `~/.mltk/user_settings.yaml`
11. Download any other logs files
12. Execute any shutdown shell commands configured in model specification and/or `~/.mltk/user_settings.yaml`


## Settings

The following settings are used by the `mltk ssh` command.
Note that most of these settings are optional and may be configured
in several different locations (see the next section, "Settings Locations", for more details).


### Remote Working Directory

This is the working directory where the MLTK command will execute.  
- This setting is __optional__
- Default: `.`

This setting can be specified in one of three locations (in order of priority):

1. The `--host` command-line option, e.g.  
   `mltk ssh --host my_server.com/workspace`
2. The `SshMixin` model mixin property, e.g.  
   `my_model.remote_dir = '~/workspace`
3. The `ssh.remote_dir` setting in `~/.mltk/user_settings.yaml`, e.g.  
   ```yaml
   ssh: 
     remote_dir:  ~/workspace
   ```

### Hostname

The name of the SSH server. This can be:  
- Domain name of server, e.g.: myserver.com
- IP address, e.g.: 145.243.23.222
- Host name in [~/.ssh/config](https://linuxize.com/post/using-the-ssh-config-file/)
  
This setting is __required__.

This setting can be specified in one of three locations (in order of priority):

1. The `hostname` in [~/.ssh/config](https://linuxize.com/post/using-the-ssh-config-file)
   that maps to the `Host` provided on the command-line with the `--host` option, e.g.:  
   `mltk ssh --host my_server`  -> find `Host` entry in `~/.ssh/config` with name `my_server` -> Use corresponding `Hostname` value
2. The `--host` command-line option, e.g.  
   `mltk ssh --host myserver.com`
3. The `ssh.connection.hostname` setting in `~/.mltk/user_settings.yaml`, e.g.  
   ```yaml
   ssh: 
     connection:
       hostname:  myserver.com
   ```

### Port

The listening port of the SSH server.   
- This setting is __optional__
- Default: `22`

This setting can be specified in one of four locations (in order of priority):

1. The `--port` command-line option, e.g.:  
   `mltk ssh --host ssh3.vast.ai -p 34567` -> port=34567
2. The `--host` command-line option, e.g.  
   `mltk ssh --host ssh3.vast.ai:34567` -> port=34567
3. The `User` in [~/.ssh/config](https://linuxize.com/post/using-the-ssh-config-file)
   that maps to the `Host` provided on the command-line with the `--host` option, e.g.:  
   `mltk ssh --host my_server`  -> find `Host` entry in `~/.ssh/config` with name `my_server`  -> Use corresponding `Port` value
4. The `ssh.connection.port` setting in `~/.mltk/user_settings.yaml`, e.g.  
   ```yaml
   ssh: 
     connection:
       port: 2222
   ```


### Username

The SSH login username.  
- This setting is __optional__

This setting can be specified in one of three locations (in order of priority):

1. The `--host` command-line option, e.g.  
   `mltk ssh --host root@ssh3.vast.ai:34567` -> username=root
2. The `User` in [~/.ssh/config](https://linuxize.com/post/using-the-ssh-config-file)
   that maps to the `Host` provided on the command-line with the `--host` option, e.g.:  
   `mltk ssh --host my_server`  -> find `Host` entry in `~/.ssh/config` with name `my_server` -> Use corresponding `User` value
3. The `ssh.connection.username` setting in `~/.mltk/user_settings.yaml`, e.g.  
   ```yaml
   ssh: 
     connection:
       username: root
   ```


### Key Filename

The filepath to the SSH [private key](https://wiki.archlinux.org/title/SSH_keys).  
- This setting is __optional__

This setting can be specified in one of three locations (in order of priority):

1. The `-i` command-line option, e.g.  
   `mltk ssh myserver.com -i ~/.ssh/id_myserver` 
2. The `IdentityFile` in [~/.ssh/config](https://linuxize.com/post/using-the-ssh-config-file)
   that maps to the `Host` provided on the command-line with the `--host` option, e.g.:  
   `mltk ssh --host my_server`  -> find `Host` entry in `~/.ssh/config` with name `my_server` -> Use corresponding `IdentityFile` value
3. The `ssh.connection.key_filename` setting in `~/.mltk/user_settings.yaml`, e.g.  
   ```yaml
   ssh: 
     connection:
       key_filename: ~/.ssh/id_myserver
   ```

### Environment

List _or_ dictionary of environment variables to export before executing MLTK command on remote server.  
- This setting is __optional__

This setting can be specified in two locations (in order, higher is __merged__ with lower (so higher overwrites lower)):

1. The `SshMixin` model mixin property, e.g.  
   `my_model.environment = ['PROD_ENV=1', 'CUDA_DEVICES=2']` _or_   
   `my_model.environment = dict(PROD_ENV=1, CUDA_DEVICES=2)`
2. The `ssh.environment` setting in `~/.mltk/user_settings.yaml`, e.g.  
   ```yaml
   ssh: 
     environment:
     - PROD_ENV=1
     - CUDA_DEVICES=
   ```
   or  
   ```yaml
   ssh: 
     environment:
       PROD_ENV: 1
       CUDA_DEVICES: 2
   ```

### Upload Files

List of file paths to upload from the local to remote before executing MLTK command.
- This setting is __optional__

If the path does _not_ contain a pipe `|`, e.g.: `dataset/*.csv`, then:
- The local path is relative to the model specification script
- The remote path is relative to the remote working directory
- Absolute paths are _not_ allowed
- The path may use the recursive [glob](https://docs.python.org/3/library/glob.html) format

If the path _does_ contain a pipe `|`, e.g.: `~/patch.txt|./patch.txt`, then:
- Format is `<local path>|<remote path>`
- The local path is relative to the model specification script
- The remote path is relative to the remote working directory
- Both paths may be absolute
- _No_ wildcards


This setting can be specified in two locations (in order, higher is __appended__ lower):

1. The `SshMixin` model mixin property, e.g.  
   `my_model.upload_files = ['dataset.zip', 'dataset/*.csv']`
2. The `ssh.upload_files` setting in `~/.mltk/user_settings.yaml`, e.g.  
   ```yaml
   ssh: 
     upload_files:
     - dataset.zip
     - dataset/*.csv
     - ~/patch.txt|./patch.txt
   ```

### Startup Commands

List of shell commands to execute on remote machine before executing the MLTK command.
- This setting is __optional__
- The commands run in a [bash shell](https://en.wikipedia.org/wiki/Bash_(Unix_shell))

This setting can be specified in two locations (in order, higher is __appended__ lower):

1. The `SshMixin` model mixin property, e.g.  
   `my_model.startup_cmds = ['pip install mylib', 'sudo apt-get install 7zip']`
2. The `ssh.startup_cmds` setting in `~/.mltk/user_settings.yaml`, e.g.  
   ```yaml
   ssh: 
     startup_cmds:
     - pip install mylib
     - sudo apt-get install 7zip
   ```

### Download Files

List of file paths to download from the remote to local after executing MLTK command.
- This setting is __optional__


If the path does _not_ contain a pipe `|`, e.g.: `logs/*.txt`, then:
- The local path is relative to the model specification script
- The remote path is retlavie to the remote working directory
- Absolute paths are _not_ allowed
- The path may use the recursive [glob](https://docs.python.org/3/library/glob.html) format

If the path _does_ contain a pipe `|`, e.g.: `./results.txt|~/results.txt`, then:
- Format is `<remote path>|<local path>`
- The local path is relative to the model specification script
- The remote path is relative to the remote working directory
- Both paths may be absolute
- _No_ wildcards


This setting can be specified in two locations (in order, higher is __appended__ lower):

1. The `SshMixin` model mixin property, e.g.  
   `my_model.download_files = ['results.zip', 'logs/*.txt']`
2. The `ssh.download_files` setting in `~/.mltk/user_settings.yaml`, e.g.  
   ```yaml
   ssh: 
     download_files:
     - results.zip
     - logs/*.txt
     - ./results.txt|~/results.txt
   ```

### Shutdown Commands

List of shell commands to execute on remote machine after executing the MLTK command.
- This setting is __optional__
- The commands run in a [bash shell](https://en.wikipedia.org/wiki/Bash_(Unix_shell))

This setting can be specified in two locations (in order, higher is __appended__ lower):

1. The `SshMixin` model mixin property, e.g.  
   `my_model.shutdown_cmds = ['curl -F data=log.txt my_server.com']`
2. The `ssh.shutdown_cmds` setting in `~/.mltk/user_settings.yaml`, e.g.  
   ```yaml
   ssh: 
     shutdown_cmds:
     - curl -F `data=log.txt` my_server.com
   ```


## Settings Locations

The various settings may be specified in the following locations:

### Command-line options

There are three command-line options:

#### --host

```shell
mltk ssh --host [<user name>@]<host>[:<port>][/<path>]
```

Where:  
- `<user name>` - Optional, user login name
- `<host>` - Required, SSH hostname or name in [~/.ssh/config](https://linuxize.com/post/using-the-ssh-config-file)
- `<port>` - Optional, SSH port, default is 22
- `<path>` - Optional, remote directory path

Examples:  
- `mltk ssh --host my_server`
- `mltk ssh --host myserver.com`
- `mltk ssh --host 192.168.1.56`
- `mltk ssh --host ubuntu@192.168.1.56`
- `mltk ssh --host ubuntu@192.168.1.56:456`
- `mltk ssh --host ubuntu@192.168.1.56/workspace`


#### --port

```shell
mltk ssh --port <port> 
```

Where:
- `<port>` is the SSH server's listening port


#### --identity_file

```
mltk ssh --identity_file <file path> 
```

Where:
- `<file path>`- Is the file path to the SSH private key


### SshMixin

The [SshMixin](https://siliconlabs.github.io/mltk/docs/python_api/mltk_model/ssh_mixin.html) model mixin allows for defining model-specific SSH settings.
 
__NOTE:__ This mixin is __optional__, it is _not_ required to run the model with the `ssh` command.

#### Example

```python

# Import MLTK model object and mixins
from mltk.core import (
    MltkModel,
    TrainMixin,
    AudioDatasetMixin,
    EvaluateClassifierMixin,
    SshMixin,
)

# Instantiate MltkModel with SshMixin
class MyModel(
    MltkModel, 
    TrainMixin, 
    AudioDatasetMixin, 
    EvaluateClassifierMixin,
    SshMixin
):
    pass
my_model = MyModel()

# Define model-specific SSH properties
my_model.ssh_remote_dir = '~/workspace'
my_model.ssh_create_venv = True
my_model.ssh_environment = ['DEV=1', 'CUDA_DEVICES=2']
my_model.ssh_startup_cmds = ['pip install mylib']
my_model.ssh_upload_files = ['dataset.zip', 'dataset/*.csv']
my_model.ssh_download_files = ['results.zip']
my_model.ssh_shutdown_cmds = ['echo "all done"']

```


### ~/.mltk/user_settings.yaml

The [user_settings.yaml](https://siliconlabs.github.io/mltk/docs/other/settings_file.html) file allows for defining user-specific MLTK settings.  
This file must be manually created at `~/.mltk/user_settings.yaml`.

The following SSH settings may be added to this file (all settings are __optional__):

```yaml
ssh:
    config_path: <path to ssh config file on local machine>
    remote_dir: <path to remote working directory>
    create_venv: <true/false, if a MLTK python venv should be automatically created on the remote machine>
    connection:
        hostname: <SSH server hostname>
        port: <SSH server listening port>
        username: <user login name>
        key_filename: <path to private key on local machine>
    environment: <list of environment variables to export on remote machine>
    upload_files: <list of files to upload to remote machine>
    startup_cmds: <list of shell commands to execute on remote machine before executing MLTK command>
    download_files: <list of files to download after executing MLTK command>
    shutdown_cmds: <list of shell commands to execute after executing MLTK command>

```

#### Example


File: `~/.mltk/user_settings.yaml`:  

```yaml
ssh:
    config_path: ~/ssh_config
    remote_dir: ~/workspace
    create_venv: false
    connection:
        hostname: my_server.com
        port: 222
        username: joe
        key_filename: ~/.ssh/id_my_server
    environment:
    - CUDA_VISIBLE_DEVICES=-1
    - DEV_ENV=1
    upload_files:
    - dataset.zip
    - config.txt
    startup_cmds:
    - pip install silabs-mltk
    - sudo apt install -y p7zip-full libsndfile1
    download_files:
    - custom_logs/**
    shutdown_cmds:
    - curl -F `data=log.txt` my_server.com
```

### ~/.ssh/config

The [SSH Config](https://linuxize.com/post/using-the-ssh-config-file/) file is a standard file used by the SSH client.  
By default, this file is located at `~/.ssh/config`. This path can be overridden by defining the `ssh.config_path` setting
in `~/.mltk/user_settings.yaml`, e.g.:  

```yaml
ssh:
   config_path: custom/path/ssh/config
```

Refer to the online documentation for more details about the contents of this file: [SSH Config](https://linuxize.com/post/using-the-ssh-config-file/)


#### Example

File: `~/.ssh/config`:  

```
Host vast_ai
  HostName ssh6.vast.ai
  Port 31521
  User root
  StrictHostKeyChecking no
  IdentityFile ~/.ssh/id_vast_ai
```

Then issuing the following command will use the config file settings:

```shell
mltk ssh --host vast_ai train image_example1
```


## Command Examples

Executing MLTK commands on a remote machine via SSH is done using the `ssh` operation.

For more details on the available command-line options, issue the command:

```shell
mltk ssh --help
```

When a command is invoked, it executes in a detached sub-process. This way, if the SSH connection
prematurely disconnects, the command will continue to execute.

Issuing `Ctrl+C` will abort the command on both the local and remote machines.


The following are examples of how remote SSH training can be invoked from the command-line:


### Example 1: Train with settings configured in user_settings.yaml

The following shows how to train the `keyword_spotting_on_off_v2` model on a remote server.  
In this example, all of the SSH settings are configured in the `~/.mltk/user_settings.yaml`.  
After training completes, the results are downloaded to the local machine.

```shell
mltk ssh train keyword_spotting_on_off_v2
```


### Example 2: Train with settings on command-line

The following shows to train the `keyword_spotting_on_off_v2` model on a remote server.  
In this example, the SSH server settings are provided on the command-line.  
After training completes, the results are downloaded to the local machine.

```shell
mltk ssh -h root@ssh5.vast.ai/workspace -p 23452 -i ~/.ssh/id_vast_ai train keyword_spotting_on_off_v2
```

The `-h` option has the following format: `[<user name>@]<host>[:<port>][/<path>]`  
where:
- `<user name>` - user login name (optional)
- `<host>` - SSH server hostname
- `<port>` - SSH server listening port (optional)
- `<path>` - Remote directory path (optional)

The `-p` is the SSH server's listening port.  
And the `-i` option points to the SSH [private key](https://wiki.archlinux.org/title/SSH_keys) file on the local machine.


### Example 3: Train without wait for results

The following shows to train the `keyword_spotting_on_off_v2` model on a remote server.  
In this example, the SSH server hostname is provided and the login info is retrieved from the `~/.ssh/config` file.
Since the `--no-wait` option is provided, the command does _not_ wait for the training to complete on the remote server.
Instead, the command immediately returns and the training command executes on the remote server in the background.
To retrieve the training results, the `--resume` option was be later provided (see Example 4 below).

```shell
mltk ssh -h vast.ai train keyword_spotting_on_off_v2 --no-wait
```

### Example 4: Retrieve results from previous training session

The following shows how to retrieve the results of a previously executed command.  
This is useful if SSH connection prematurely disconnects or the `--no-wait` option was previously used.

This will wait until the previously invoked command has completed on the remote server then 
download the training results.

```shell
mltk ssh -h vast.ai train keyword_spotting_on_off_v2 --resume
```

__HINT:__ You could also use the `--no-wait` option to poll the remote server to see if the command has completed without waiting for it to finish.


### Example 5: Train new model, and forcefully discard previous

Only one command may be active on the remote server. The `--force` option may be used to abort a previously invoked command.

```shell
mltk ssh train keyword_spotting_on_off_v2 --force
```
