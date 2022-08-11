# Settings File

Additional MLTK settings may be configure in the settings file:

```shell
~/.mltk/user_settings.yaml
```

Where `~` is your OS's user home directory.  
On Windows, this typically points to `C:\Users\<user name>` where `<user name>` is your Windows user name.


__NOTE:__ The environment variable `MLTK_USER_SETTINGS_PATH` may be used to override the default path `~/.mltk/user_settings.yaml`:


This file uses the [YAML](https://www.tutorialspoint.com/yaml/yaml_quick_guide.htm) format.


## Example 

An example `~/.mltk/user_settings.yaml` is as follows:

```yaml
model_paths:
    - ~/dev_models
    - c:/production_models

commander:
    device: efr32
    serial_number: 123432

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
    - sudo apt install -y p7zip-full libsndfile1
    download_files:
    - custom_logs/**
    shutdown_cmds:
    - curl -F `data=log.txt` my_server.com
```


## Settings

The following user settings are supported:


### model_paths

This is a list of directories to search of MLTK models.  
See [Model Search Path](../guides/model_search_path.md) for more details.


```{note}
Environment variables used in the directory paths will be resolved at runtime
```


### commander

[Simplicity Commander](https://www.silabs.com/documents/public/user-guides/ug162-simplicity-commander-reference-guide.pdf) settings.
This is useful if more than one development board is connected to the local computer.  

The following sub-settings are supported:  
 - `device` - The device code given to the `--device` command-line option
 - `serial_number` - The adapter serial number given to the `--serialno` command-line option
 - `ip_address` - The adapter IP address given to the `--ip` command-line option


### ssh

These are settings specific to the [mltk ssh](../guides/model_training_via_ssh.md) command.
See the [Model Training via SSH](../guides/model_training_via_ssh.md) guide for more details.