# Model Search Path

Most commands/APIs support providing a model name argument (as opposed to a file path). e.g:

```shell
mltk train image_example1
mltk profile my_model
mltk evaluate keyword_spotting_model
```

```python
mltk_model = load_mltk_model('image_example1')
train_model('image_example1')
profile_model('my_model')
evaluate_model('keyword_spotting_model')
```


The MLTK searches the following paths for a model specification script `.py` and/or model archive file `.mltk.zip` with a matching name:

1. __model_paths in user_settings.yaml__  
    The [user_settings.yaml](../other/settings_file.md) file may contain the field `model_paths` which should be a list of directories containing model specification scripts.
    If the setting field exists, these directories are recursively searched in the provided order.
2. __Current working directory__  
    The current working directory (_not_ including the MLTK Python package/repository root directory) is searched (__not__ recursively).
3. __Environment Variable: MLTK_MODEL_PATHS__  
    The environment variable [MLTK_MODEL_PATHS](../other/environment_variables.md) may contain a list of model search directories.
    If the environment variable exists, these directories are recursively searched in the provided order.
4. __MLTK package module: mltk.models__  
    The default search path is the `mltk.models` module directory which is recursively searched.