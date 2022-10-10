import os
import sys
import threading
import logging
from mltk.utils.logger import get_logger as _get_logger
from mltk.utils.logger import make_filelike
from mltk.utils.shell_cmd import run_shell_cmd
from mltk.utils.path import create_user_dir
from mltk.core.model.model_utils import find_model_specification_file

_lock = threading.Lock()

pytest_results_dir = create_user_dir('pytest_results')



def get_logger(name='utest', console=False):
    logger = _get_logger(
        name, 
        level='DEBUG', 
        console=console, 
        log_file=f'{pytest_results_dir}/{name}.log'
    )
    make_filelike(logger)
    return logger


def run_mltk_command(*args, update_model_path=False, logger=None, env=None, exe_path=None) -> str:
    if logger is None:
        logger_name = args[0]
        if logger_name.startswith('-'):
            logger_name = 'mltk'
        logger = get_logger(f'{logger_name}_cli_tests')
    env = env or os.environ.copy()

    python_bin_dir = os.path.dirname(os.path.abspath(sys.executable))
    exe_path = exe_path or os.path.join(python_bin_dir, 'mltk')

    env['MLTK_UNIT_TEST'] = '1'
    cmd = [exe_path]
    cmd.extend(args)
    cmd_str = ' '.join([str(x) for x in cmd])

    logger.info('\n' + '*' * 100)
    logger.info(f'Running {cmd_str}')

    with _lock:
        if update_model_path:
            env['MLTK_MODEL_PATHS'] = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
        retcode, retmsg = run_shell_cmd(cmd, outfile=logger, env=env, logger=logger)
        if retcode not in (0, 2):
            raise RuntimeError(f'Command failed (err code={retcode}): {cmd_str}\n{retmsg}')
    return retmsg
        

def generate_run_model_params(
    train=True,
    evaluate=None,
    profile=True,
    summarize=None,
    quantize=None,
    view=None,
    build=None,
    tflite=None,
    run_directly=True
):
    use_basic_model_params = os.environ.get('MLTK_UTEST_BASIC_MODEL_PARAMS', '0') == '1'
    if use_basic_model_params:
        evaluate = evaluate if evaluate is not None else False 
        summarize = summarize if summarize is not None else False 
        quantize = quantize if quantize is not None else False 
        view = view if view is not None else False 
        build = build if build is not None else False
    else:
        evaluate = evaluate if evaluate is not None else True 
        summarize = summarize if summarize is not None else True 
        quantize = quantize if quantize is not None else True 
        view = view if view is not None else True 


    names = ['op', 'tflite', 'build']
    params = []
    tflite_str = 'tflite' if tflite else 'no_tflite'
    build_str = 'build' if build else 'no_build'

    if run_directly:
        params.append(('run-directly', 'na', 'na'))

    if train:
        params.append(('train', 'na', 'na'))

    if evaluate:
        if tflite is None:
            params.append(('evaluate', 'no_tflite', 'na'))
            params.append(('evaluate', 'tflite',    'na'))
        else:
            params.append(('evaluate', tflite_str, 'na'))

    if profile:
        if build is None:
            params.append(('profile', 'na', 'no_build'))
            params.append(('profile', 'na', 'build'))
        else:
            params.append(('profile', 'na', build_str))
        
    if summarize:
        if build is None and tflite is None:
            params.append(('summarize', 'no_tflite', 'no_build'))
            params.append(('summarize', 'no_tflite', 'build'))
            params.append(('summarize', 'tflite',    'no_build'))
            params.append(('summarize', 'tflite',    'build'))
        elif build is not None and tflite is None: 
            params.append(('summarize', 'no_tflite', build_str))
            params.append(('summarize', 'tflite',   build_str))
        elif build is None and tflite is not None: 
            params.append(('summarize', tflite_str, 'no_build'))
            params.append(('summarize', tflite_str, 'build'))
        else:
            params.append(('summarize', tflite_str, build_str))
        
    if quantize:
        params.append(('quantize', 'na', 'na'))

    if view:
        if build is None and tflite is None:
            params.append(('view', 'no_tflite', 'no_build'))
            params.append(('view', 'no_tflite', 'build'))
            params.append(('view', 'tflite',    'no_build'))
            params.append(('view', 'tflite',    'build'))
        elif build is not None and tflite is None: 
            params.append(('view', 'no_tflite', build_str))
            params.append(('view', 'tflite',   build_str))
        elif build is None and tflite is not None: 
            params.append(('view', tflite_str, 'no_build'))
            params.append(('view', tflite_str, 'build'))
        else:
            params.append(('view', tflite_str, build_str))

    return names, params


def run_model_operation(
    name_or_archive:str, 
    op:str,
    tflite=None,
    build=None
):
    if tflite == 'tflite':
        tflite = True 
    elif tflite == 'no_tflite' or tflite == 'na':
        tflite = False
    if build == 'build':
        build = True 
    elif build == 'no_build' or build == 'na':
        build = False


    logger = get_logger('model_operation_tests')
    logger.info(f'Testing {name_or_archive}, op={op}, tflite={tflite}, build={build}')
    name_no_test = name_or_archive.replace('-test', '')
    update_archive_arg = '--update-archive' if name_or_archive.endswith('-test') else '--no-update-archive'

    if op == 'run-directly':
        model_spec_path = find_model_specification_file(name_or_archive)
        run_mltk_command(model_spec_path, logger=logger, exe_path=sys.executable) 

    elif op == 'train':
        run_mltk_command('train', name_or_archive, '--clean', '--verbose', '--test', logger=logger)
    
    elif op == 'evaluate':
        run_mltk_command('evaluate', name_or_archive, '--verbose', update_archive_arg, logger=logger)
    
    elif op == 'profile':
        if build:
            run_mltk_command('profile', name_or_archive, '--build', '--verbose', logger=logger)
        elif not name_or_archive.endswith('-test'):
            run_mltk_command('profile', name_or_archive, '--verbose', logger=logger)
        
    elif op == 'summarize':
        if build:
            if tflite:
                run_mltk_command('summarize', name_no_test, '--tflite', '--build', '--verbose', logger=logger)
            else:
                run_mltk_command('summarize', name_no_test, '--build', '--verbose', logger=logger)
        elif not name_or_archive.endswith('-test'):
            if tflite:
                run_mltk_command('summarize', name_or_archive, '--tflite', '--verbose', logger=logger)
            else:
                run_mltk_command('summarize', name_or_archive, '--verbose', logger=logger)

    elif op == 'quantize':
        run_mltk_command('quantize', name_or_archive, '--verbose', update_archive_arg, logger=logger)

    elif op == 'view':
        if build:
            if tflite:
                run_mltk_command('view', name_no_test, '--tflite', '--build', '--verbose', logger=logger)
            else:
                run_mltk_command('view', name_no_test, '--build', '--verbose', logger=logger)
        elif not name_or_archive.endswith('-test'):     
            if tflite:
                run_mltk_command('view', name_or_archive, '--tflite', '--verbose', logger=logger)
            else:
                run_mltk_command('view', name_or_archive, '--verbose', logger=logger)

    else:
        raise ValueError(f'Unknown test op: {op}')