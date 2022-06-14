import sys
import os
import typer
import pytest
from contextlib import redirect_stdout

from mltk import cli
from mltk.utils.path import clean_directory


@cli.root_cli.command('utest')
def utest_command(
    test_type: str = typer.Argument(None, 
        help='''\b
Comma separated list of unit test types, options are:
- all - Run all the tests, default if omitted
- cli - Run CLI tests
- api - Run API tests
- model - Run reference model tests
- cpp - Build C++ applications tests
- studio - Simplicity Studio project generation/building tests
'''
    ),
    test_arg: str = typer.Argument(None, 
        help='Argument for specific test(s) to run. Refer to the pytests -k option for more details: https://docs.pytest.org/en/latest/example/markers.html#using-k-expr-to-select-tests-based-on-their-name'
    ),
    verbose: bool = typer.Option(False, '--verbose', '-v', 
        help='Enable verbose console logs'
    ),
    list_only: bool = typer.Option(False, '--list', '-l', 
        help='Only list the available unit tests'
    ),
    clear_cache: bool = typer.Option(True, 
        help='Clear the MLTK cache directory before running tests'
    ),
):
    """Run the all unit tests"""

    # Import all required packages here instead of at top
    # to help improve the CLI's responsiveness
    from mltk import MLTK_DIR, MLTK_ROOT_DIR
    from mltk.utils.python import as_list
    from mltk.utils.test_helper import get_logger, logger_dir
    from mltk.utils.path import create_tempdir, clean_directory

    all_types = {'all', 'cli', 'api', 'model', 'cpp', 'studio'}
    test_type = test_type or 'all'
    test_type = set(test_type.split(','))

    if not test_type.issubset(all_types):
        cli.abort(msg=f'Unsupported test type: {",".join(test_type)}. Supported types are: {",".join(all_types)}')


    test_dirs = []
    if test_type & {'all', 'cli'}:
        test_dirs.append('cli/tests')

    if test_type & {'all', 'api'}:
        test_dirs.append('core/tflite_micro/tests')
        test_dirs.append('core/tflite_model/tests')
        test_dirs.append('core/tflite_model_parameters/tests')
        test_dirs.append('core/preprocess/audio/audio_feature_generator/tests')

    if test_type & {'all', 'model'}:
        test_dirs.append('models/tests')

    if test_type & {'all', 'cpp'}:
        test_dirs.append('../cpp/tools/tests/test_build_apps.py')
        test_dirs.append('../cpp/tools/tests/test_build_cli.py')

    if test_type & {'all', 'studio'}:
        test_dirs.append('../cpp/tools/tests/test_simplicity_studio.py')


    test_dirs = as_list(test_dirs)

    clean_directory(logger_dir)
    logger = get_logger('utest_cli', console=True)
    logger.set_terminator('')
    logger.info(f'Generating logs at: {logger_dir}\n')

    if clear_cache:
        utest_cache_dir = create_tempdir('utest_cache')
        clean_directory(utest_cache_dir)
        os.environ['MLTK_CACHE_DIR'] = utest_cache_dir
        logger.warning(f'Setting MLTK_CACHE_DIR="{os.environ["MLTK_CACHE_DIR"]}"\n')
    else:
        logger.warning(f'NOT clearing MLTK cache, using existing cache at ~/.mltk\n')
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Disable the GPU as well
    # cli.print_info('Setting CUDA_VISIBLE_DEVICES=-1')

    cmd = []
    cmd.append(f'--rootdir={MLTK_ROOT_DIR}')
    cmd.append(f'--html-report={logger_dir}/report.html')
    cmd.append('--color=yes')
    cmd.extend(['-o', 'log_cli=true'])
    if verbose:
        cmd.append('-v')
        cmd.append('--log-cli-level=DEBUG')
    cmd.append('--show-capture=all')
    cmd.extend(['-W', 'ignore::DeprecationWarning'])
    if list_only:
        cmd.append('--collect-only')
    if test_arg is not None:
        cmd.extend(['-k', test_arg])

    for d in test_dirs:
        cmd.append(f'{MLTK_DIR}/{d}')
    

    logger.info('Executing: pytest ' + " ".join(cmd) + '\n')
    with redirect_stdout(logger):
        retcode = pytest.main(cmd)
    
    logger.info(f'\n\nFor more details, see: {logger_dir}\n')
    cli.abort(code=retcode)