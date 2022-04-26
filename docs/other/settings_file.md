# Settings File

Additional MLTK settings may be configure in the settings file:

```shell
~/.mltk/user_settings.yaml
```

Where `~` is your OS's user home directory.

The file uses the [YAML](https://www.tutorialspoint.com/yaml/yaml_quick_guide.htm) format.

## Example 

An example `~/.mltk/user_settings.yaml` is as follows:

```yaml
model_paths:
    - ~/dev_models
    - c:/production_models

commander:
    device: efr32
    serial_number: 123432
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

