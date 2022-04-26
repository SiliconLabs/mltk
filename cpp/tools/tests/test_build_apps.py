import pytest

from mltk.utils import cmake
from mltk.utils.system import get_current_os
from mltk.utils.test_helper import get_logger
from mltk.utils.path import create_tempdir

#SPECIFIC_PLATFORM = 'brd4186'


APP_TARGETS = [ 
    'mltk_hello_world',
    'mltk_audio_classifier',
    'mltk_image_classifier',
    'mltk_model_profiler',
    'mltk_fingerprint_authenticator'
]


EMBEDDED_PLATFORMS = [ 
    'brd2601',
    'brd2204',
    'brd4166',
    'brd4186',
]
ALL_PLATFORMS = [
    get_current_os(),
    *EMBEDDED_PLATFORMS
]

PLATFORM_ACCELERATORS = {
    'brd2601' : 'mvp',
    'brd4186': 'mvp',
    'windows': 'mvp',
    'linux': 'mvp'
}


build_params = []
def _add_app(target, platforms, accelerators=None):
    accelerators = accelerators or {}
    for p in platforms:
        if 'SPECIFIC_PLATFORM' in globals() and p != globals()['SPECIFIC_PLATFORM']:
            continue

        build_params.append((target, p, None))
        if p in accelerators:
            build_params.append((target, p, accelerators[p]))


_add_app('mltk_hello_world', ALL_PLATFORMS)
_add_app('mltk_model_profiler', ALL_PLATFORMS, PLATFORM_ACCELERATORS)
_add_app('mltk_audio_classifier', [get_current_os(), 'brd2601'])
_add_app('mltk_image_classifier', EMBEDDED_PLATFORMS)
_add_app('mltk_fingerprint_authenticator', EMBEDDED_PLATFORMS)


app_build_logger = get_logger('build_app_tests')


@pytest.mark.parametrize(['target', 'platform', 'accelerator'], build_params)
def test_build_app(target, platform, accelerator):
    build_dir = create_tempdir(f'utest/build_app/{target}-{platform}-{accelerator}')
    cmake.build_mltk_target( 
        target=target,
        platform=platform,
        build_dir=build_dir,
        build_subdir=False,
        accelerator=accelerator,
        logger=app_build_logger,
        clean=True,
    )