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


## MLTK_READONLY

Set this variable to `1` to indicate that the MLTK is running on a "read-only" file-system.  
This is useful if te MLTK package is running in a cloud "lambda" function.  

When set, the MLTK will only write to the OS's temporary directory.

