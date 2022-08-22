# Environment Variables

The MLTK uses the following _optional_ environment variables:

## MLTK_MODEL_PATHS

This should be a list of directory paths to search for MLTK models.  
Each directory path should be delimited with the OS's path delimiter

- __Windows__ - Semicolon `;`
- __Linux__ - Colon `:`

See [Model Search Path](../guides/model_search_path) for more details.

## MLTK_CACHE_DIR

Specify the directory path to the MLTK's cache directory.  
If omitted, the MLTK defaults to the directory: `~/.mltk`

## MLTK_USER_SETTINGS_PATH

Specify the path to the [user_settings.yaml](./settings_file.md).  
If omitted, the settings file points to `~/.mltk/user_settings.yaml`.

## MLTK_READONLY

Set this variable to `1` to indicate that the MLTK is running on a "read-only" file-system.  
This is useful if te MLTK package is running in a cloud "lambda" function.  

When set, the MLTK will only write to the OS's temporary directory.


## MLTK_SETUP_PY_DEPS

This is used by [setup.py](https://github.com/SiliconLabs/mltk/blob/master/setup.py), the script used to install the MLTK Python package.
This may be used to override the MLTK Python package dependencies.

This should contain a list of [install requirements](https://packaging.python.org/en/latest/discussions/install-requires-vs-requirements/) delimited by a pipe `|`, e.g.:

```
export MLTK_SETUP_PY_DEPS="tensorflow==2.4.4|numpy==1.19.5|tflite-support==0.2.0|tensorflow_probability==0.12.2|onnxruntime==1.9.0|typing-extensions==3.7.4"
```
