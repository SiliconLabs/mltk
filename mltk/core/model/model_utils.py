from typing import List, Union
import os
import logging
import re
import shutil
import collections

from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.models import load_model as load_keras_model

from mltk import __version__ as mltk_version_str
from mltk import MLTK_ROOT_DIR
from mltk import models as mltk_models
from mltk.core.utils import get_mltk_logger
from mltk.utils.path import (fullpath, walk_with_depth, create_tempdir, get_user_setting)
from mltk.utils.python import as_list, import_module_at_path, prepend_exception_msg
from mltk.utils import gpu

from .model import MltkModel
from .mixins.archive_mixin import (
    ARCHIVE_EXTENSION,
    TEST_ARCHIVE_EXTENSION,
    get_archive_extension,
    extract_file
)
from ..tflite_model import TfliteModel




def load_mltk_model(
    model: str,
    test:bool=False,
    print_not_found_err=False,
    logger: logging.Logger = None,
    reload:bool = True
) -> MltkModel:
    """Find a MLTK model with the given name and instantiate its corresponding :py:class:`mltk.core.MltkModel` object

    Args:
        model: Name of MLTK model or path to MLTK model's python specification script or archive.
            Append `-test` to the MLTK model name to load into "testing" mode (this is the same as setting the argument: test=True)
        test: If the MLTK model should be loaded in "testing" mode
        print_not_found_err: If true and the MLTK model is not found, then print an error
        reload: If the given model is a python script then reload the module if necessary

    Returns:
        Load model object
    """
    from mltk.cli import print_did_you_mean_error # pylint: disable=import-outside-toplevel

    logger = logger or get_mltk_logger()

    if not isinstance(model, str):
        raise Exception('Model argument must be a string')

    # Resolve any path variables if applicable
    model_path = fullpath(model)

    # If a model file path was given
    if os.path.exists(model_path):
        return load_mltk_model_with_path(model_path, test=test, logger=logger, reload=reload)
    elif model_path.endswith('.mltk.zip'):
        raise FileNotFoundError(f'Model archive not found: {model_path}')
    elif model_path.endswith(('.tflite', '.h5')):
        raise FileNotFoundError(f'Model file not found: {model_path}')
    elif model_path.endswith('.py'):
        raise FileNotFoundError(f'Model specification script not found: {model_path}')

    # Otherwise the name of an MLTK model was given
    # So attempt to find it on the search paths

    # If '-test' was appended to the MLTK model name
    # then load the model in "test" mode
    if model.endswith('-test'):
        test = True
        model = model[:-len('-test')]

    if not re.match(r'^[a-zA-Z0-9_]+$', model, re.DOTALL):
        raise ValueError('Invalid MLTK model argument given. Must either be the path to an existing model file (.tflite, .h5, .mltk.zip) or must contain only letters, numbers, or an underscore')

    logger.debug(f'Searching for MLTK model: {model}')
    model_spec_path = find_model_specification_file(
        model=model,
        test=test,
        logger=logger,
        print_not_found_err=print_not_found_err
    )
    if not model_spec_path:
        raise Exception(f'Failed to find model specification file with name: {model}.py')

    return load_mltk_model_with_path(
        model_path=model_spec_path,
        test=test,
        logger=logger
    )


def load_mltk_model_with_path(
    model_path: str,
    test:bool=False,
    logger: logging.Logger=None,
    reload:bool = True
) -> MltkModel:
    """Instantiate a MltkModel object from the given model path
    The model path can be a ``.py`` model specificaton or a model archive ``.mltk.zip``.
    """
    if not model_path.endswith(('.py', ARCHIVE_EXTENSION)):
        raise Exception('Model path must have either .mltk.zip or .py extension')

    logger = logger or get_mltk_logger()

    # Resolve any path variables if applicable
    model_path = fullpath(model_path)

    model_base, _ = os.path.splitext(model_path)
    model_base = model_base.replace('\\', '/')
    model_name = os.path.basename(model_base)
    model_spec_path = f'{model_base}.py'

    # If the path to a model archive was given
    # then copy the archive to a temp directory
    # and extract the model specification file from the archive
    if model_path.endswith(ARCHIVE_EXTENSION):
        model_name = os.path.basename(model_path).replace(TEST_ARCHIVE_EXTENSION, '').replace(ARCHIVE_EXTENSION, '')
        temp_dir = create_tempdir(f'tmp_model_specs/{model_name}')
        shutil.copy(model_path, temp_dir)

        logger.info(f'Extracting {model_name}.py from {model_path}')
        model_spec_path = extract_file(
            archive_path=f'{temp_dir}/{os.path.basename(model_path)}',
            name=f'{model_name}.py',
            dest_dir=temp_dir
        )

        if model_path.endswith(TEST_ARCHIVE_EXTENSION):
            test = True

    try:
        logger.debug(f'Importing {model_spec_path}')
        model_module = import_module_at_path(model_spec_path, reload=reload)
    except Exception as e:
        prepend_exception_msg(e, f'Failed to import MLTK model module: {model_spec_path}')
        raise

    for key in dir(model_module):
        mltk_model = getattr(model_module, key)
        if isinstance(mltk_model, MltkModel):
            mltk_version = _parse_version(mltk_version_str)
            # Issue a warning if the model's mltk version is different than the current mltk version
            # This can happen when a model archive is generated with an older version of the mltk
            model_mltk_version_str = getattr(model_module, '__mltk_version__', None)
            if model_mltk_version_str is not None:
                model_mltk_version = _parse_version(model_mltk_version_str)
                if model_mltk_version.major != mltk_version.major or model_mltk_version.minor != mltk_version.minor:
                    logger.warning(f'Model {mltk_model.name} was created with mltk version: {model_mltk_version_str} but current mltk version is: {mltk_version_str}')

            if test:
                mltk_model.enable_test_mode()
            return mltk_model

    raise Exception(f'Model specification file: {model_spec_path} does not define a MltkModel object')


def load_tflite_or_keras_model(
    model: Union[MltkModel, str],
    model_type:str=None,
    weights: str=None,
    logger: logging.Logger=None
) -> Union[TfliteModel, KerasModel]:
    """Instantiate a Keras or TfliteModel object

    IF model is an :py:class:`mltk.core.MltkModel` instance OR a model archive `.mltk.zip`,
    AND model_type is:

    - ``None`` -> return built :py:class:`mltk.core.KerasModel` from model specification
    - ``tflite`` -> return loaded :py:class:`mltk.core.TfliteModel` from model archive
    - ``h5`` -> return loaded :py:class:`mltk.core.KerasModel` from model archive

    ELSE model should be the file path to a `.tflite` or `.h5` model file.
    """

    from .mixins.train_mixin import TrainMixin

    logger = logger or get_mltk_logger()

     # Initialize the GPU if necessary
    if (isinstance(model, MltkModel) and model_type is None) \
        or (isinstance(model, str) and model.endswith(('.h5', '.mltk.zip'))):
        gpu.initialize(logger=logger)


    if isinstance(model, MltkModel) or (isinstance(model, str) and model.endswith('.mltk.zip')):
        if isinstance(model, str) and model.endswith('.mltk.zip'):
            model = load_mltk_model(model)

        if model_type is None:
            if not isinstance(model, TrainMixin):
                raise Exception('MltkModel must inherit TrainMixin')
            logger.debug('Building Keras model')

            # Ensure test mode is enabled
            test_mode_enabled = model.attributes['test_mode_enabled']
            model.attributes['test_mode_enabled'] = True

            built_model = model.build_model_function(model)
            if built_model is None:
                raise RuntimeError('Your "my_model.build_model_function" must return the compiled Keras model (did you forget to add the "return keras_model" statement at the end?')
            elif not isinstance(built_model, KerasModel):
                raise RuntimeError('Your "my_model.build_model_function" must return the compiled Keras model instance')

            on_save_model = getattr(model, 'on_save_keras_model', None)
            if on_save_model is not None:
                try:
                    built_model = on_save_model(
                        mltk_model=model,
                        keras_model=built_model,
                        logger=logger
                    )
                except Exception as e:
                    prepend_exception_msg(e, 'Error while calling my_model.on_save_keras_model')
                    raise

            # Restore whatever the test mode state was
            model.attributes['test_mode_enabled'] = test_mode_enabled

        elif model_type in ('h5', '.h5', 'keras'):
            h5_path = model.h5_archive_path

            try:
                logger.debug(f'Loading Keras model from {model.archive_path}')
                built_model = load_keras_model(h5_path, custom_objects=model.keras_custom_objects)
            except Exception as e:
                prepend_exception_msg(e, 'Failed to load Keras .h5 file')
                raise

        elif model_type in ('tflite', '.tflite'):
            tflite_path = model.tflite_archive_path

            try:
                logger.debug(f'Loading .tflite model from {model.archive_path}')
                built_model = TfliteModel.load_flatbuffer_file(tflite_path)
            except Exception as e:
                prepend_exception_msg(e, 'Failed to load .tflite file')
                raise

        else:
            raise Exception('archive_extension must be h5, tflite or None')

    elif isinstance(model, str):
        if model.endswith('.h5'):
            try:
                logger.debug(f'Loading Keras model from {model}')
                built_model = load_keras_model(model)
            except Exception as e:
                prepend_exception_msg(e, 'Failed to load Keras .h5 file')
                raise

        elif model.endswith('.tflite'):
            try:
                logger.debug(f'Loading .tflite model from {model}')
                built_model = TfliteModel.load_flatbuffer_file(model)
            except Exception as e:
                prepend_exception_msg(e, 'Failed to load .tflite file')
                raise

        else:
            raise Exception('Must provide path to .h5 or .tflite model file')

    else:
        raise Exception('model must be a str or MltkModel')


    if weights:
        if isinstance(built_model, KerasModel):
            weights_file = weights if isinstance(model, str) else model.get_weights_path(weights)
            logger.info(f'Loading weights: {weights_file}')
            built_model.load_weights(weights_file)

        else:
            logger.warning('Loading weights into .tflite model not supported')


    return built_model


def load_tflite_model(
    model: Union[str, MltkModel, TfliteModel],
    build:bool=False,
    print_not_found_err:bool=False,
    return_tflite_path:bool=False,
    test:bool=False,
    logger: logging.Logger=None
) -> Union[TfliteModel,str]:
    """Return the path to a .tflite model file or a TfliteModel instance

    Args:
        model: One of the following:
            - An MltkModel model instance
            - An TfliteModel model instance
            - The path to a .tflite
            - The path to a .mltk.zip model archive
            - The path to a .py MLTK model specification
            - The name of an MLTK model
        build: If the given Mltk model should be built into a .tflite
        print_not_found_err: If the model model is not found, print possible alternatives and exit
        return_tflite_path: If true, return the file path to the .tflite, otherwise return a TfliteModel instance
        test: If a "test" model is provided
        logger: Optional logger

    Return:
        The corresponding TfliteModel if return_tflite_path=False or the path to the .tflite if return_tflite_path=True
    """

    logger = logger or get_mltk_logger()
    mltk_model:MltkModel = None
    tflite_model:TfliteModel = None
    model_name = None

    if isinstance(model, MltkModel):
        mltk_model = model
        model_name = mltk_model.name
        model = mltk_model.tflite_archive_path


    if isinstance(model, TfliteModel):
        if build:
            raise RuntimeError('Cannot use build option with TfliteModel instance')
        tflite_model = model
        model_name = (tflite_model.filename or 'my_model.tflite')[:-len('.tflite')]

    elif isinstance(model, str):
        if build and model.endswith(('.tflite', '.mltk.zip')):
            raise RuntimeError('Cannot use --build option with .tflite or .mltk.zip model argument. Must be model name or path to model specification (.py)')
        elif model.endswith('.h5'):
            raise ValueError('Must provide .tflite or .mltk.zip model file type')

        if model.endswith('.tflite'):
            model = fullpath(model)
            if return_tflite_path:
                return model

            tflite_model = TfliteModel.load_flatbuffer_file(model)
            model_name = tflite_model.filename[:-len('.tflite')]

        elif not model.endswith('.mltk.zip'):
            if build:
                mltk_model = load_mltk_model(
                    model,
                    test=test,
                    logger=logger,
                    print_not_found_err=print_not_found_err
                )

            else:
                model_spec_path = find_model_specification_file(
                    model=model,
                    test=test,
                    logger=logger,
                    print_not_found_err=print_not_found_err
                )
                if model_spec_path is None:
                    raise ValueError(f'Failed to find model specification file with name: {model}.py')

                model = model_spec_path[:-len('.py')]
                if test:
                    model += '-test'
                model += '.mltk.zip'


        if model.endswith('.mltk.zip'):
            from .mixins.archive_mixin import extract_file

            model_name = os.path.basename(model[:-len('.mltk.zip')])
            if model_name.endswith('-test'):
                model_name = model_name[:-len('-test')]
                tflite_name = f'{model_name}.test.tflite'
            else:
                 tflite_name = f'{model_name}.tflite'
            tflite_path = extract_file(model, tflite_name)
            if return_tflite_path:
                return tflite_path

            tflite_model = TfliteModel.load_flatbuffer_file(tflite_path)

    if build:
        from ..quantize_model import quantize_model

        if mltk_model is None:
            raise RuntimeError('Must provide MltkModel instance, name of MltkModel, other .py path to model specification to use the build option')

        logger.info('--build option provided, building model rather than using trained model')
        tflite_model = quantize_model(
            model=mltk_model,
            build=True,
            output='tflite_model'
        )


    if return_tflite_path:
        tflite_path = create_tempdir('tmp_models') + f'/{model_name}.tflite'
        tflite_model.save(tflite_path)
        return tflite_path

    else:
        assert tflite_model is not None
        return tflite_model


def list_mltk_models(
    test:bool=False,
    for_utests=False,
    logger:logging.Logger=None
) -> List[str]:
    """Return a list of all found MLTK model names"""

    logger = logger or get_mltk_logger()

    found_models = []
    search_dirs = _get_model_search_dirs()
    archive_ext = get_archive_extension(test=False)
    test_archive_ext = get_archive_extension(test=True)

    mltk_model_re = re.compile(r'.*\s@mltk_model\s.*')
    utest_disable_re = re.compile(r'.*\s@mltk_utest_disabled\s.*')

    def _process_python_file(py_path):
        retval = False
        with open(py_path, 'r') as f:
            for line in f:
                if for_utests and utest_disable_re.match(line):
                    return False

                if mltk_model_re.match(line):
                    retval = True
                    if not for_utests:
                        break

        return retval


    for search_dir in search_dirs:
        for root, _, files in walk_with_depth(search_dir, depth=5, followlinks=True):
            for fn in files:
                if fn.endswith('.py'):
                    try:
                        p = f'{root}/{fn}'.replace('\\', '/')
                        if _process_python_file(p):
                            found_models.append(fn[:-len('.py')])
                    except Exception as e:
                        logger.warning(f'Failed to process Python file: {p}, err: {e}')

                if test:
                    if fn.endswith(test_archive_ext):
                        found_models.append(fn.replace(test_archive_ext, ''))
                else:
                    if fn.endswith(archive_ext) and not fn.endswith(test_archive_ext):
                        found_models.append(fn.replace(archive_ext, ''))

            # Do NOT recurse into the CWD
            if search_dir == os.curdir:
                break


    return sorted(set(found_models))


def find_model_specification_file(
    model:str,
    test:bool=False,
    logger:logging.Logger=None,
    print_not_found_err:bool=False
) -> str:
    """Given the model name, attempt to find its corresponding python specification file.
    The specification file could be in a model archive.
    """
    logger = logger or get_mltk_logger()
    search_dirs = _get_model_search_dirs()
    cwd = fullpath(os.getcwd())

    if model.endswith('-test'):
        test = True
        model = model[:-len('-test')]
    if model.endswith('.py'):
        model = fullpath(model)

    model_subdir = os.path.dirname(model)
    model_name, _ = os.path.splitext(os.path.basename(model))

    py_path = None
    archive_path = None
    if model_subdir:
        model_subdir = f'{model_subdir}/'

    archive_ext = get_archive_extension(test=test)
    model_path = f'/{model_subdir}{model_name}.py'
    model_arc_path = f'{model_subdir}/{model_name}{archive_ext}'

    logger.debug(f'Model search path(s): {",".join(search_dirs)}')
    for search_dir in search_dirs:
        if py_path is not None:
            break
        for root, _, files in os.walk(search_dir, followlinks=True):
            root = root.replace('\\', '/')
            for fn in files:
                file_path = f'{root}/{fn}'

                if file_path.endswith(model_path):
                    py_path = file_path
                if file_path.endswith(model_arc_path):
                    archive_path = file_path

            # If the spec was found then break out of the loop
            if py_path is not None:
                break
            # Do NOT recurse into the CWD
            if search_dir == cwd:
                break

    if py_path is None and archive_path is not None:
        logger.info(f'Extracting {model_name}.py from {archive_path}')
        py_path = extract_file(
            archive_path=archive_path,
            name=f'{model_name}.py',
            dest_dir=os.path.dirname(archive_path)
        )

    if not py_path and print_not_found_err:
        from mltk.cli import print_did_you_mean_error # pylint: disable=import-outside-toplevel
        all_models = list_mltk_models(test=test)
        print_did_you_mean_error('Failed to find model', model, all_models, and_exit=True)

    return py_path



def _get_model_search_dirs() -> List[str]:
    """Return list of model search directories

    This populates the list as follows:
    - ~/.mltk/user_settings.yaml:model_paths
    - CWD
    - MLTK_MODEL_PATHS OS environment variable
    - mltk.models package directory
    """
    search_dirs = as_list(get_user_setting('model_paths'))

    # Include the CWD only if it's not the root of the mltk repo
    curdir = fullpath(os.getcwd())
    if fullpath(MLTK_ROOT_DIR) != curdir:
        search_dirs.append(os.getcwd())

    env_paths = os.getenv('MLTK_MODEL_PATHS', '')
    if env_paths:
        search_dirs.extend(env_paths.split(os.pathsep))

    search_dirs.append(os.path.dirname(mltk_models.__file__))

    search_dirs = [fullpath(x) for x in search_dirs]

    return search_dirs

_Version = collections.namedtuple('_Version', ['major', 'minor', 'patch'])


def _parse_version(version):
    toks = version.split('.')
    major = 0 if len(toks) < 1 else int(toks[0])
    minor = 0 if len(toks) < 2 else int(toks[1])
    patch = 0 if len(toks) < 3 else int(toks[2])
    return _Version(major, minor, patch)