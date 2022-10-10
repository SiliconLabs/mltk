import os
import sys 
import webbrowser
import re
import shutil
import typer
import logging

from mltk import cli



from mltk import MLTK_ROOT_DIR
from mltk.utils.shell_cmd import run_shell_cmd
from mltk.utils.python import install_pip_package
from mltk.utils.path import (fullpath, clean_directory, remove_directory, get_user_setting)



@cli.build_cli.command('docs')
def build_docs_command(
    show: bool = typer.Option(True,
        help='View the docs HTML after building'
    ),
    verbose: bool = typer.Option(False, '-v', '--verbose',
        help='Enable verbose logging'
    ),
    checklinks: bool = typer.Option(True,
        help='Check that all the links work in the generated HTML'
    ),
    clean: bool = typer.Option(True,
        help='Clean the build directory before building'
    ),
    revert_only: bool = typer.Option(False,
        help='Only reset the generated docs (i.e. do a git revert on the docs directory)'
    ),
):
    """Build the MLTK online documentation
    
    This uses the sphinx docs build system to convert the
    markdown files in <mltk repo>/docs into a website
    """
    logger = cli.get_logger(verbose=True)

    if revert_only:
        _revert_docs_dir(logger)
        return



    install_pip_package('sphinx==4.5.0', logger=logger)
    install_pip_package('myst-parser==0.17.2', 'myst_parser', logger=logger)
    install_pip_package('myst-nb==0.15.0', 'myst_nb', logger=logger)
    install_pip_package('numpydoc==1.3.1', logger=logger)
    install_pip_package('sphinx_autodoc_typehints==1.18.2', logger=logger)
    install_pip_package('sphinx-markdown-tables==0.0.17', 'sphinx_markdown_tables', logger=logger)
    install_pip_package('sphinx-copybutton==0.5.0', 'sphinx_copybutton', logger=logger)
    install_pip_package('sphinx-panels==0.6.0', 'sphinx_panels', logger=logger)
    install_pip_package('nbclient==0.5.13', 'nbclient', logger=logger)
    
    install_pip_package('git+https://github.com/linkchecker/linkchecker.git', 'linkcheck', logger=logger)
    install_pip_package('git+https://github.com/bashtage/sphinx-material.git', 'sphinx_material', logger=logger)

    _patch_sphinx_autosummary_generate_py()
    
    repo_project_name = get_user_setting('repo_project_name', 'siliconlabs')

    env = os.environ.copy()
    env['MLTK_REPO_PROJECT_NAME'] = repo_project_name

    sphinx_exe = os.path.dirname(sys.executable) + '/sphinx-build'

    docs_dir = f'{MLTK_ROOT_DIR}/docs/website_builder'
    source_dir = f'{docs_dir}/source'
    build_dir = f'{docs_dir}/build'

    built_files = [ 
        'genindex.html',
        'index.html',
        'py-modindex.html',
        'search.html',
        'searchindex.js'
    ]
    built_dirs = [ 
        '_images',
        '_modules',
        '_sources',
        '_static',
        '_downloads',
        'docs',
        'mltk',
    ]
    website_builder_build_dirs = [ 
        'jupyter_execute',
        'build',
        'source/docs',
        'source/mltk',
    ]


    if clean:
        logger.info(f'Cleaning {build_dir}')
        clean_directory(build_dir)
        for fn in built_files:
            p =  f'{MLTK_ROOT_DIR}/docs/{fn}'
            if os.path.exists(p):
                os.remove(p)
        for dn in built_dirs:
            remove_directory(f'{MLTK_ROOT_DIR}/docs/{dn}')
        for dn in website_builder_build_dirs:
            remove_directory(f'{MLTK_ROOT_DIR}/docs/website_builder/{dn}')

    cmd = [
        sphinx_exe, 
    ]
    if verbose:
        cmd.append('-v')

    cmd.extend([ '-b', 'html', source_dir, build_dir])

    retcode, retmsg = run_shell_cmd(cmd, outfile=logger, env=env, logger=logger)
    if retcode != 0:
        cli.abort(msg=f'Failed to build docs: {retmsg}')

    # Copy the generated docs from build directory
    # to the docs directory
    # Updating the "siliconlabs" to te test repo's name as necessary
    repo_name = get_user_setting('repo_project_name')
    for fn in built_files:
        _copy_file(f'{build_dir}/{fn}', f'{MLTK_ROOT_DIR}/docs/{fn}', repo_name=repo_name)
    for dn in built_dirs:
        _copy_directory(f'{build_dir}/{dn}', f'{MLTK_ROOT_DIR}/docs/{dn}', repo_name=repo_name)


    if show:
        webbrowser.open(f'file:///{MLTK_ROOT_DIR}/docs/index.html')

    if checklinks:
        linkchecker_exe = os.path.dirname(sys.executable) + '/linkchecker'
        if os.name == 'nt':
            linkchecker_exe += '.exe'
        index_path = os.path.abspath(fullpath(f'{MLTK_ROOT_DIR}/docs/index.html'))
        retcode, _ = run_shell_cmd([
            sys.executable, linkchecker_exe, 
            '--check-extern', 
            '--ignore-url', r'.*assets\.slid\.es.*', 
            '--ignore-url', r'.*assets-v2\.slid\.es.*', 
            '--ignore-url', r'.*linuxize\.com.*', 
            '--ignore-url', r'.*timeseriesclassification\.com.*', 
            '--ignore-url', r'http\:\/\/localhost.*',
            index_path], 
            outfile=logger,
            logger=logger
        )
        if retcode != 0:
            cli.abort(msg='Invalid links detected')


def _copy_file(src, dst, repo_name=None):
    url_re = re.compile(r'.*="(https:\/\/siliconlabs\.github\.io\/mltk\/).*\..*".*', re.I)

    if not src.endswith(('.html', '.txt')):
        shutil.copy(src, dst)
        return 

    # Ensure all absolute docs URLs are relative
    is_html = src.endswith('.html')
    dst_dir = os.path.dirname(dst)
    docs_base_dir = f'{MLTK_ROOT_DIR}/docs'
    with open(src, 'r', encoding='utf-8') as f:
        data = ''
        for line in f:
            if is_html:
                match = url_re.match(line)
                if match:
                    relpath = os.path.relpath(docs_base_dir, dst_dir).replace('\\', '/')
                    line = line.replace(match.group(1), relpath + '/')
            data += line 

    # If a testing repo name was given
    # then update any URLs found in the html docs files
    if repo_name:
        data = data.replace('siliconlabs.github.io/mltk', f'{repo_name}.github.io/mltk')
        data = data.replace('SiliconLabs.github.io/mltk', f'{repo_name}.github.io/mltk')
        data = data.replace('github/siliconlabs/mltk', f'github/{repo_name}/mltk')
        data = data.replace('github/SiliconLabs/mltk', f'github/{repo_name}/mltk')
        data = data.replace('github.com/siliconlabs/mltk', f'github.com/{repo_name}/mltk')
        data = data.replace('github.com/SiliconLabs/mltk', f'github.com/{repo_name}/mltk')

    with open(dst, 'w', encoding='utf-8') as f:
        f.write(data)


def _copy_directory(src, dst, repo_name=None):
    os.makedirs(dst, exist_ok=True)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            _copy_directory(s, d, repo_name=repo_name)
        else:
            _copy_file(s, d, repo_name=repo_name)

def _revert_docs_dir(logger:logging.Logger):
    logger.warning('Reverting all generated docs')

    def _clean_dir(path:str):
        run_shell_cmd(
            ['git', 'restore', '--source=HEAD', '--staged', '--worktree', '--', f'{MLTK_ROOT_DIR}/docs/{path}'],
            logger=logger
        )
        run_shell_cmd(
            ['git', 'clean', '-fd', f'{MLTK_ROOT_DIR}/docs/{path}'],
            logger=logger
        )

    _clean_dir('docs')
    _clean_dir('mltk')
    _clean_dir('_images')
    _clean_dir('_modules')
    _clean_dir('_sources')
    _clean_dir('_static')
    _clean_dir('_downloads')

    run_shell_cmd(
        ['git', 'checkout', 'HEAD', '*.html'],
        cwd=f'{MLTK_ROOT_DIR}/docs',
        logger=logger
    )

    run_shell_cmd(
        ['git', 'checkout', 'HEAD', '*.js'],
        cwd=f'{MLTK_ROOT_DIR}/docs',
        logger=logger
    )



def _patch_sphinx_autosummary_generate_py():
    """This patches:
    <python site packages>/sphinx/ext/autosummary/generate.py

    It makes it so the generated auto summary file is generated in given table-of-contents file.

    e.g.:
    
    .. autosummary::
       :toctree: audio_data_generator_params
       :template: custom-class-template.rst

    Then the auto summary will be generated in <directory containing rst>/audio_data_generator_params.rst

    """
    from sphinx.ext.autosummary import generate

    data = ''
    updated = False
    with open(generate.__file__, 'r') as f:
        for line in f:
            if 'filename = os.path.join(path, filename_map.get(name, name) + suffix)' in line.strip():
                line = line.replace('filename = os.path.join(path, filename_map.get(name, name) + suffix)', 'filename = path + suffix # Patched by the MLTK # os.path.join(path, filename_map.get(name, name) + suffix)')
                updated = True
            data += line
    
    if updated:
        with open(generate.__file__, 'w') as f:
            print(f'Patched {generate.__file__}')
            f.write(data)