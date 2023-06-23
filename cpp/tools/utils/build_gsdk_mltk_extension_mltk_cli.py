import re
import os
import webbrowser
import yaml
import typer
from mltk import cli, MLTK_ROOT_DIR
from mltk.utils.cmake import build_mltk_target




@cli.build_cli.command("gsdk_mltk_extension")
def build_gsdk_mltk_extension_command(
    show: bool = typer.Option(True,
        help='Show video demonstrating how to add the MLTK to Simplicity Studio 5'
    ),
):
    """Build and install the MLTK extension into Silicon Lab's Gecko SDK

    This allows for building and running the MLTK example C++ applications
    from the Simplicity Studio 5 development environment
    """

    logger = cli.get_logger(verbose=True)
    gsdk_mltk_dir = f'{MLTK_ROOT_DIR}/cpp/shared/gecko_sdk'

    try:
        _update_mltk_dependencies(logger=logger)
    except Exception as e:
        cli.handle_exception('Failed to update MLTK C++ dependencies', e)


    try:
        gsdk_dir, git_tag = _get_gsdk_dir(gsdk_mltk_dir, logger=logger)
    except Exception as e:
        cli.handle_exception('Failed to get GSDK repo directory', e)

    try:
        _update_gsdk_dir(gsdk_dir, git_tag, logger=logger)
    except Exception as e:
        cli.handle_exception('Failed to update GSDK repo', e)

    try:
        _update_gsdk_properties_file(gsdk_dir, logger=logger)
    except Exception as e:
        cli.handle_exception('Failed to update GSDK .properties', e)

    try:
        _update_gsdk_slcs_file(gsdk_dir, logger=logger)
    except Exception as e:
        cli.handle_exception('Failed to update GSDK gecko_sdk.slcs', e)

    try:
        _create_mltk_symlink(gsdk_dir, logger=logger)
    except Exception as e:
        cli.handle_exception('Failed to create symlink to MLTK', e)

    try:
        _update_apps_xml(gsdk_mltk_dir, logger=logger)
    except Exception as e:
        cli.handle_exception('Failed to update apps.xml', e)

    logger.info('\nImport the "MLTK Gecko SDK Suite" into Simplicity Studio')
    gif_path = f'{MLTK_ROOT_DIR}/docs/img/ss_import_mltk.gif'
    logger.info(f'See: {gif_path}\n')
    if show:
        webbrowser.open_new_tab(gif_path)

    logger.info('From Simplicity Studio 5,')
    logger.info('1. On top toolbar, click: Window')
    logger.info('2. Preferences')
    logger.info('3. Simplicity Studio')
    logger.info('4. SDKs')
    logger.info('5. Add SDK...')
    logger.info(f'6. Update the Location to: {os.path.normpath(gsdk_dir)}')
    logger.info('7. Click: OK')
    logger.info('8. If prompted, click the "Trust" button')
    logger.info('9. Click: Apply and Close')
    logger.info('\nAt this point, the "MLTK Gecko SDK Suite" is now available in Simplicity Studio 5')
    logger.info('From the Launcher, select your connected device, then select the "Preferred SDK" to be: MLTK Gecko SDK Suite')
    logger.info('The MLTK example applications should now be available for project creation.')


def _update_mltk_dependencies(logger):
    """Clone all the embedded C++ dependencies"""
    logger.info('Updating C++ dependencies')
    build_mltk_target(
        mltk_target='mltk_audio_classifier',
        target='mltk_tflite_micro_apply_patch',
        platform='brd2601',
        clean=True
    )

def _update_gsdk_properties_file(gsdk_dir, logger):
    """Update the .properties file to rename:
    "Gecko SDK Suite" to "MLTK Gecko SDK Suite"
    """
    properties_path = f'{gsdk_dir}/.properties'

    logger.info(f'Updating {properties_path}')
    s = ''
    line_ending = None
    with open(properties_path, 'r') as f:
        for line in f:
            if line_ending is None:
                line_ending = line.replace(line.rstrip(), '')
            if line.startswith('label='):
                line = f'label=MLTK Gecko SDK Suite{line_ending}'
            elif line.startswith('description='):
                line = f'description=MLTK Gecko SDK Suite{line_ending}'
            elif line.startswith('extendedProperties='):
                if 'mltk.properties' not in line:
                    line = 'extendedProperties=extension/mltk/mltk.properties ' + line[len('extendedProperties='):]
            s += line

    with open(properties_path, 'w') as f:
        f.write(s)


def _update_gsdk_slcs_file(gsdk_dir, logger):
    """Update the gecko_sdk.slsc file to rename:
    "Gecko SDK Suite" to "MLTK Gecko SDK Suite"
    """
    gecko_sdk_slcs_path = f'{gsdk_dir}/gecko_sdk.slcs'

    logger.info(f'Updating {gecko_sdk_slcs_path}')
    s = ''
    line_ending = None
    with open(gecko_sdk_slcs_path, 'r') as f:
        for line in f:
            if line_ending is None:
                line_ending = line.replace(line.rstrip(), '')
            if line.startswith('label: '):
                line = f'label: "MLTK Gecko SDK Suite"{line_ending}'
            s += f'{line}'

    with open(gecko_sdk_slcs_path, 'w') as f:
        f.write(s)


def _create_mltk_symlink(gsdk_dir, logger):
    """Create hardlink:
    <mltk dir>/cpp/shared/gecko_sdk/<version>/extension/mltk <-> <mltk dir>/cpp
    """
    os.makedirs(f'{gsdk_dir}/extension', exist_ok=True)

    dst_dir = os.path.normpath(f'{gsdk_dir}/extension/mltk')
    src_dir = os.path.normpath(os.path.realpath(f'{MLTK_ROOT_DIR}/cpp'))

    logger.info(f'Creating symlink: {src_dir} <-> {dst_dir}')
    if not os.path.exists(dst_dir):
        if os.name == 'nt':
            import _winapi
            _winapi.CreateJunction(src_dir, dst_dir)
        else:
            import subprocess
            subprocess.check_call(['ln', '-s', src_dir, dst_dir])


def _update_apps_xml(gsdk_mltk_dir, logger):
    """Update apps.xml for each MLTK example app"""
    updated_lines = False
    project_path_re = re.compile(r'<properties key="projectFilePaths" value="([\w\.\/]+)"\/>')
    apps_xml_path = f'{gsdk_mltk_dir}/simplicity_studio/apps.xml'
    apps_xml_dir = os.path.dirname(apps_xml_path)

    with open(apps_xml_path, 'r') as f:
        app_xml_lines = f.read().splitlines(keepends=True)

        app_start = -1
        project_path = None
        for lineno, line in enumerate(app_xml_lines):
            if app_start == -1:
                if line.strip().startswith('<descriptors'):
                    app_start = lineno
                continue

            if project_path is None:
                project_path_match = project_path_re.match(line.strip())
                if project_path_match is not None:
                    project_path = os.path.normpath(f'{apps_xml_dir}/{project_path_match.group(1)}')
                continue

            if line.strip().startswith('</descriptors>'):
                if _update_app_xml_entry(app_xml_lines, app_start, lineno, project_path):
                    updated_lines = True
                app_start = -1
                project_path = None


    if updated_lines:
        with open(apps_xml_path, 'w') as f:
            for line in app_xml_lines:
                f.write(line)


def _update_app_xml_entry(lines, start, end, project_path):
    """Update an entry in apps.xml by copying the info from a .slcp project file to the apps.xml file"""
    updated = False

    def _xml_escape( str ):
        str = str.replace("&", "&amp;")
        str = str.replace("<", "&lt;")
        str = str.replace(">", "&gt;")
        str = str.replace("\"", "&quot;")
        str = str.replace("\n", "&#10;")
        str = str.replace("\r", "&#13;")
        return str

    def _set_line(lineno, value):
        nonlocal updated
        if lines[start + lineno] != value:
            updated = True
            lines[start + lineno] = value

    with open(project_path, 'r') as f:
        project_data = yaml.load(f, Loader=yaml.SafeLoader)


    description = _xml_escape(project_data['description'])
    _set_line(0, f'  <descriptors name="{project_data["project_name"]}" label="{project_data["label"]}" description="{description}">\n')

    for lineno, line in enumerate(lines[start:end]):
        if line.strip().startswith('<properties key="category"'):
            _set_line(lineno, f'    <properties key="category" value="{project_data["category"]}" />\n')
        if line.strip().startswith('<properties key="quality"'):
            _set_line(lineno, f'    <properties key="quality" value="{project_data["quality"]}" />\n')

    return updated


def _get_gsdk_dir(gsdk_mltk_dir, logger) -> str:
    """Get the path to the GSDK repo downloaded by the MLTK build scripts
    """
    cmakeliststxt_path = f'{gsdk_mltk_dir}/CMakeLists.txt'

    cache_version_re = re.compile(r'\s*CACHE_VERSION\s+([\w\.]+)\s*.*')
    git_tag_re = re.compile(r'\s*GIT_TAG\s+([\w\.]+)\s*.*')

    sdk_version = None
    git_tag = None
    with open(cmakeliststxt_path, 'r') as f:
        for line in f:
            match = cache_version_re.match(line)
            if match:
                sdk_version = match.group(1)
            match = git_tag_re.match(line)
            if match:
                git_tag = match.group(1)

    if not (sdk_version and git_tag):
        raise RuntimeError(f'Failed to parse {cmakeliststxt_path} for GSDK version and GIT tag')


    gsdk_dir = f'{gsdk_mltk_dir}/{sdk_version}'
    logger.info(f'GSDK path: {gsdk_dir}')

    return gsdk_dir, git_tag



def _update_gsdk_dir(gsdk_dir, git_tag, logger):
    """Update the GSDK repo by either doing a git clone or git lfs pull"""
    from mltk.utils.shell_cmd import run_shell_cmd

    if not os.path.exists(f'{gsdk_dir}/.git'):
        gsdk_git_url = 'https://github.com/SiliconLabs/gecko_sdk.git'
        logger.info(f'Cloning Gecko SDK from:\n{gsdk_git_url}\nto: {gsdk_dir}\nNOTE: This may take awhile as all of the GSDK binary assets need to be downloaded ...')

        version_dir = os.path.basename(gsdk_dir)
        retcode, retmsg = run_shell_cmd(['git', 'clone', gsdk_git_url, version_dir], cwd=os.path.dirname(gsdk_dir), outfile=logger)
        if retcode  != 0:
            raise RuntimeError(f'Failed to clone {gsdk_git_url} to {gsdk_dir}')

        retcode, retmsg = run_shell_cmd(['git', 'checkout', git_tag], cwd=gsdk_dir, outfile=logger)
        if retcode != 0:
            raise RuntimeError(f'Failed to checkout {git_tag} from  {gsdk_dir}')

    else:
        logger.info(f'Updating {gsdk_dir}\nNOTE: This may take awhile as all of the GSDK binary assets need to be downloaded ...')
        retcode, retmsg = run_shell_cmd(['git', 'lfs', 'pull'], cwd=gsdk_dir, outfile=logger)
        if retcode  != 0:
            raise RuntimeError(f'Failed to update {gsdk_dir}')

