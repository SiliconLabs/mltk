import os
import sys 
import webbrowser
import shutil
import typer

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
):
    """Build the MLTK online documentation
    
    This uses the sphinx docs build system to convert the
    markdown files in <mltk repo>/docs into a website
    """
    logger = cli.get_logger(verbose=True)


    install_pip_package('sphinx==4.2.0', logger=logger)
    install_pip_package('myst-parser', 'myst_parser', logger=logger)
    install_pip_package('myst-nb', 'myst_nb', logger=logger)
    install_pip_package('numpydoc', logger=logger)
    install_pip_package('sphinx_autodoc_typehints', logger=logger)
    install_pip_package('sphinx-markdown-tables', 'sphinx_markdown_tables', logger=logger)
    install_pip_package('sphinx-copybutton', 'sphinx_copybutton', logger=logger)
    install_pip_package('sphinx-panels', 'sphinx_panels', logger=logger)
    
    install_pip_package('git+https://github.com/linkchecker/linkchecker.git', 'linkcheck', logger=logger)
    install_pip_package('git+https://github.com/bashtage/sphinx-material.git', 'sphinx_material', logger=logger)

    
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

    retcode, _ = run_shell_cmd(cmd, outfile=logger, env=env)
    if retcode != 0:
        cli.abort(msg='Failed to build docs')

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
            index_path], 
            outfile=logger,
            logger=logger
        )
        if retcode != 0:
            cli.abort(msg='Invalid links detected')


def _copy_file(src, dst, repo_name=None):
    if not src.endswith(('.html', '.txt')):
        shutil.copy(src, dst)
        return 

    with open(src, 'r', encoding='utf-8') as f:
        data = f.read()

    # If a testing repo name was given
    # then update any URLs found in the html docs files
    if repo_name:
        data = data.replace('siliconlabs.github.io/mltk', f'{repo_name}.github.io/mltk')
        data = data.replace('github/siliconlabs/mltk', f'github/{repo_name}/mltk')
        data = data.replace('github.com/siliconlabs/mltk', f'github.com/{repo_name}/mltk')

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
