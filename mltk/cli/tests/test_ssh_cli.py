
import pytest
import os
import stat
import yaml

# Skip these tests if there is no SSH server locally running on Linux
pytestmark = pytest.mark.skipif(not os.path.exists('/var/run/sshd.pid'), reason='No SSH server found or not running on Linux')

from mltk.utils.shell_cmd import run_shell_cmd
from mltk.utils.path import create_tempdir, fullpath, create_user_dir, remove_directory
from mltk.utils.test_helper import run_mltk_command, pytest_results_dir, get_logger
from mltk.utils import test_helper

TEST_MODELS_DIR = os.path.dirname(os.path.abspath(test_helper.__file__)).replace('\\', '/')


logger = get_logger()
remote_working_dir = create_tempdir('utest/ssh_workspace')



@pytest.mark.dependency()
def test_generate_keypair():
    run_mltk_command('ssh-keygen', 'utest', '--output', pytest_results_dir)

    with open(f'{pytest_results_dir}/id_utest.pub', 'r') as f:
        pub_key = f.read()

    ssh_authorized_keys = fullpath('~/.ssh/authorized_keys')
    if os.path.exists(ssh_authorized_keys):
        with open(ssh_authorized_keys, 'r') as f:
            data = f.read()
    else:
        data = ''

    lines = [x.strip() for x in data.splitlines() if len(x) > 0]
    replace_lineno = -1
    for i, line in enumerate(lines):
        if line.endswith(' utest'):
            replace_lineno = i
            break

    if replace_lineno >= 0:
        lines[replace_lineno] = pub_key
    else:
        lines.append(pub_key)

    with open(ssh_authorized_keys + '.tmp', 'w') as f:
        f.write('\n'.join(lines))
    os.chmod(ssh_authorized_keys + '.tmp', stat.S_IRWXU)

    os.rename(ssh_authorized_keys + '.tmp', ssh_authorized_keys)


@pytest.mark.dependency(depends=['test_generate_keypair'])
def test_train_model_with_params_in_cmd_create_venv():
    startup_cmd_file_path = f'{pytest_results_dir}/test_startup_cmd.txt'
    shutdown_cmd_file_path = f'{pytest_results_dir}/test_shutdown_cmd.txt'
    model_file_path = f'{TEST_MODELS_DIR}/test_image_model-test.mltk.zip'
    model_log_dir = f'{create_user_dir()}/models/test_image_model-test'
    upload_file_abspath = f'{create_tempdir("temp")}/test_ssh_cli.py'

    ssh_workspace_dir = _get_ssh_workspace_dir()
    ssh_username = _get_ssh_username()

    _remove_file(startup_cmd_file_path)
    _remove_file(shutdown_cmd_file_path)
    _remove_file(model_file_path)
    _remove_file(upload_file_abspath)
    remove_directory(model_log_dir)
    remove_directory(ssh_workspace_dir)

    env = os.environ.copy()
    ssh_settings=dict(
        create_venv=True,
        upload_files=[f'{__file__}|{upload_file_abspath}', 'test_autoencoder_model.py'],
        environment=['TEST="This is a test"', 'FOO', 'BAR=this is a string without quotes', 'CUDA_VISIBLE_DEVICES=-1'],
        startup_cmds=["echo $TEST > ./test_startup_cmd.txt"],
        download_files=[f'./test_startup_cmd.txt|{startup_cmd_file_path}'],
        shutdown_cmds=[f'echo "shutting down ..." > {shutdown_cmd_file_path}'],
    )


    settings_path = f'{pytest_results_dir}/user_settings.yaml'
    with open(settings_path, 'w') as f:
        yaml.dump(dict(ssh=ssh_settings), f, yaml.SafeDumper)
    env['MLTK_USER_SETTINGS_PATH'] = settings_path

    run_mltk_command(
        'ssh', '-h', f'{ssh_username}@localhost/{os.path.basename(ssh_workspace_dir)}', '-p', 22, '-i', f'{pytest_results_dir}/id_utest', 'train', 'test_image_model-test', '--verbose', '--clean', '--force',
        update_model_path=True,
        env=env
    )

    assert os.path.exists(f'{ssh_workspace_dir}/test_autoencoder_model.py')
    _verify_results(
        startup_cmd_file_path=startup_cmd_file_path,
        shutdown_cmd_file_path=shutdown_cmd_file_path,
        model_file_path=model_file_path,
        model_log_dir=model_log_dir,
        upload_file_abspath=upload_file_abspath
    )



@pytest.mark.dependency(depends=['test_generate_keypair'])
def test_train_model_with_params_in_settings_no_create_venv():
    startup_cmd_file_path = f'{pytest_results_dir}/test_startup_cmd.txt'
    shutdown_cmd_file_path = f'{pytest_results_dir}/test_shutdown_cmd.txt'
    model_file_path = f'{TEST_MODELS_DIR}/test_image_model.mltk.zip'
    model_log_dir = f'{create_user_dir()}/models/test_image_model'
    upload_file_abspath = f'{create_tempdir("temp")}/test_ssh_cli.py'

    ssh_workspace_dir = _get_ssh_workspace_dir()
    ssh_username = _get_ssh_username()

    _remove_file(startup_cmd_file_path)
    _remove_file(shutdown_cmd_file_path)
    _remove_file(model_file_path)
    _remove_file(upload_file_abspath)
    remove_directory(ssh_workspace_dir)
    remove_directory(model_log_dir)

    env = os.environ.copy()
    ssh_settings=dict(
        create_venv=False,
        remote_dir=ssh_workspace_dir,
        connection=dict(
            hostname='localhost',
            port=22,
            key_filename=f'{pytest_results_dir}/id_utest',
            username=ssh_username
        ),
        upload_files=[f'{__file__}|{upload_file_abspath}', 'test_autoencoder_model.py'],
        environment=dict(TEST='This is a test', CUDA_VISIBLE_DEVICES='-1'),
        startup_cmds=[
            "echo $TEST > ./test_startup_cmd.txt",
            "python3 -m venv .venv",
            '. ./.venv/bin/activate',
            'pip3 install silabs-mltk'
        ],
        download_files=[f'./test_startup_cmd.txt|{startup_cmd_file_path}'],
        shutdown_cmds=[f'echo "shutting down ..." > {shutdown_cmd_file_path}'],
    )


    settings_path = f'{pytest_results_dir}/user_settings.yaml'
    with open(settings_path, 'w') as f:
        yaml.dump(dict(ssh=ssh_settings), f, yaml.SafeDumper)
    env['MLTK_USER_SETTINGS_PATH'] = settings_path

    run_mltk_command(
        'ssh', 'train', 'test_image_model', '--verbose', '--clean', '--force',
        update_model_path=True,
        env=env
    )

    assert os.path.exists(f'{ssh_workspace_dir}/test_autoencoder_model.py')
    _verify_results(
        startup_cmd_file_path=startup_cmd_file_path,
        shutdown_cmd_file_path=shutdown_cmd_file_path,
        model_file_path=model_file_path,
        model_log_dir=model_log_dir,
        upload_file_abspath=upload_file_abspath
    )


@pytest.mark.dependency(depends=['test_generate_keypair'])
def test_train_model_with_ssh_mixin_and_config():
    startup_cmd_file_path = f'{pytest_results_dir}/test_startup_cmd.txt'
    shutdown_cmd_file_path = f'{pytest_results_dir}/test_shutdown_cmd.txt'
    startup_cmd_file2_path = f'{pytest_results_dir}/test_startup_cmd2.txt'
    shutdown_cmd_file2_path = f'{pytest_results_dir}/test_shutdown_cmd2.txt'
    model_file_path = f'{pytest_results_dir}/test_image_model-test.mltk.zip'
    model_log_dir = f'{create_user_dir()}/models/test_image_model-test'
    upload_file_abspath = f'{create_tempdir("temp")}/test_ssh_cli.py'
    ssh_config_path = f'{pytest_results_dir}/ssh_config.txt'

    ssh_workspace_dir = _get_ssh_workspace_dir()
    ssh_username = _get_ssh_username()

    _remove_file(startup_cmd_file_path)
    _remove_file(shutdown_cmd_file_path)
    _remove_file(startup_cmd_file2_path)
    _remove_file(shutdown_cmd_file2_path)
    _remove_file(model_file_path)
    _remove_file(upload_file_abspath)
    remove_directory(ssh_workspace_dir)
    remove_directory(model_log_dir)

    with open(f'{TEST_MODELS_DIR}/test_image_model.py', 'r') as src:
        with open(f'{pytest_results_dir}/test_image_model.py', 'w') as dst:
            for line in src:
                if 'from mltk.core.model import (' in line:
                    line = f'{line}    SshMixin,\n'
                if 'class MyModel(' in line:
                    line = f'{line}    SshMixin,\n'
                dst.write(line)

            dst.write('\n\n')
            dst.write(f'mltk_model.ssh_remote_dir = "{os.path.basename(ssh_workspace_dir)}"\n')
            dst.write(f'mltk_model.ssh_create_venv = True\n')
            dst.write(f'mltk_model.ssh_environment = dict(TEST="This is a test", CUDA_VISIBLE_DEVICES="-1")\n')
            dst.write(f'mltk_model.ssh_startup_cmds = ["echo $TEST > ./test_startup_cmd.txt"]\n')
            dst.write(f'mltk_model.ssh_upload_files = ["{__file__}|{upload_file_abspath}"]\n')
            dst.write(f'mltk_model.ssh_download_files = ["./test_startup_cmd.txt|{startup_cmd_file_path}"]\n')
            dst.write(f'mltk_model.ssh_shutdown_cmds = [\'echo "shutting down ..." > {shutdown_cmd_file_path}\']\n')

    with open(ssh_config_path, 'w') as f:
        f.write('Host utest_host\n')
        f.write('  Hostname localhost\n')
        f.write('  Port 22\n')
        f.write(f'  User {ssh_username}\n')
        f.write(f'  IdentityFile {pytest_results_dir}/id_utest\n')


    env = os.environ.copy()
    ssh_settings=dict(
        config_path=ssh_config_path,
        connection=dict(
            hostname='localhost',
            port=22,
            key_filename=f'',
            username=ssh_username
        ),
        upload_files=[],
        environment=dict(TEST2='This is a test2'),
        startup_cmds=[
            "echo $TEST2 > ./test_startup_cmd2.txt",
        ],
        download_files=[f'./test_startup_cmd2.txt|{startup_cmd_file2_path}'],
        shutdown_cmds=[f'echo "shutting down2 ..." > {shutdown_cmd_file2_path}'],
    )


    settings_path = f'{pytest_results_dir}/user_settings.yaml'
    with open(settings_path, 'w') as f:
        yaml.dump(dict(ssh=ssh_settings), f, yaml.SafeDumper)
    env['MLTK_USER_SETTINGS_PATH'] = settings_path

    run_mltk_command(
        'ssh', '-h', 'utest_host', 'train', f'{pytest_results_dir}/test_image_model.py', "--test", '--verbose', '--clean', '--force',
        update_model_path=True,
        env=env
    )

    _verify_results(
        startup_cmd_file_path=startup_cmd_file_path,
        shutdown_cmd_file_path=shutdown_cmd_file_path,
        model_file_path=model_file_path,
        model_log_dir=model_log_dir,
        upload_file_abspath=upload_file_abspath,
        startup_cmd_file2_path=startup_cmd_file2_path,
        shutdown_cmd_file2_path=shutdown_cmd_file2_path,
    )


@pytest.mark.dependency(depends=['test_generate_keypair'])
def test_train_model_nowait():
    startup_cmd_file_path = f'{pytest_results_dir}/test_startup_cmd.txt'
    shutdown_cmd_file_path = f'{pytest_results_dir}/test_shutdown_cmd.txt'
    model_file_path = f'{TEST_MODELS_DIR}/test_image_model-test.mltk.zip'
    model_log_dir = f'{create_user_dir()}/models/test_image_model-test'
    upload_file_abspath = f'{create_tempdir("temp")}/test_ssh_cli.py'

    ssh_workspace_dir = _get_ssh_workspace_dir()
    ssh_username = _get_ssh_username()

    _remove_file(startup_cmd_file_path)
    _remove_file(shutdown_cmd_file_path)
    _remove_file(model_file_path)
    _remove_file(upload_file_abspath)
    remove_directory(model_log_dir)
    #remove_directory(ssh_workspace_dir)

    env = os.environ.copy()
    ssh_settings=dict(
        create_venv=True,
        upload_files=[f'{__file__}|{upload_file_abspath}', 'test_autoencoder_model.py'],
        environment=['TEST="This is a test"', 'FOO', 'BAR=this is a string without quotes', 'CUDA_VISIBLE_DEVICES=-1'],
        startup_cmds=["echo $TEST > ./test_startup_cmd.txt"],
        download_files=[f'./test_startup_cmd.txt|{startup_cmd_file_path}'],
        shutdown_cmds=[f'echo "shutting down ..." > {shutdown_cmd_file_path}'],
    )


    settings_path = f'{pytest_results_dir}/user_settings.yaml'
    with open(settings_path, 'w') as f:
        yaml.dump(dict(ssh=ssh_settings), f, yaml.SafeDumper)
    env['MLTK_USER_SETTINGS_PATH'] = settings_path

    run_mltk_command(
        'ssh', '-h', f'{ssh_username}@localhost/{os.path.basename(ssh_workspace_dir)}', '-p', 22, '-i', f'{pytest_results_dir}/id_utest', 'train', 'test_image_model-test', '--verbose', '--clean', '--force', '--no-wait',
        update_model_path=True,
        env=env
    )

@pytest.mark.dependency(depends=['test_generate_keypair', 'test_train_model_nowait'])
def test_train_model_resume():
    startup_cmd_file_path = f'{pytest_results_dir}/test_startup_cmd.txt'
    shutdown_cmd_file_path = f'{pytest_results_dir}/test_shutdown_cmd.txt'
    model_file_path = f'{TEST_MODELS_DIR}/test_image_model-test.mltk.zip'
    model_log_dir = f'{create_user_dir()}/models/test_image_model-test'
    upload_file_abspath = f'{create_tempdir("temp")}/test_ssh_cli.py'

    ssh_workspace_dir = _get_ssh_workspace_dir()
    ssh_username = _get_ssh_username()

    env = os.environ.copy()
    settings_path = f'{pytest_results_dir}/user_settings.yaml'
    env['MLTK_USER_SETTINGS_PATH'] = settings_path

    run_mltk_command(
        'ssh', '-h', f'{ssh_username}@localhost/{os.path.basename(ssh_workspace_dir)}', '-p', 22, '-i', f'{pytest_results_dir}/id_utest', 'train', 'test_image_model-test', '--verbose', '--clean', '--resume',
        update_model_path=True,
        env=env
    )

    assert os.path.exists(f'{ssh_workspace_dir}/test_autoencoder_model.py')
    _verify_results(
        startup_cmd_file_path=startup_cmd_file_path,
        shutdown_cmd_file_path=shutdown_cmd_file_path,
        model_file_path=model_file_path,
        model_log_dir=model_log_dir,
        upload_file_abspath=upload_file_abspath
    )


def _get_ssh_workspace_dir() -> str:
    # If we're running in Jenkins, then e sure to switch to the home directory
    if 'JENKINS_URL' in os.environ:
        _, retmsg = run_shell_cmd(['bash', '-l', '-c', 'cd ~ && pwd'])
    # Otherwise, just use whatever directory is used in the bash login
    else:
        _, retmsg = run_shell_cmd(['bash', '-l', '-c', 'pwd'])
    return retmsg.splitlines()[-1].strip() + '/utest_ssh_workspace'


def _get_ssh_username() -> str:
    _, retmsg = run_shell_cmd(['bash', '-l', '-c', 'id -u -n'])
    return retmsg.splitlines()[-1].strip()

def _remove_file(path):
    if os.path.exists(path):
        os.remove(path)


def _verify_results(
    startup_cmd_file_path,
    shutdown_cmd_file_path,
    model_file_path,
    model_log_dir,
    upload_file_abspath,
    startup_cmd_file2_path=None,
    shutdown_cmd_file2_path=None
):
    assert os.path.exists(startup_cmd_file_path), f'Failed to download test file: {startup_cmd_file_path}'
    with open(startup_cmd_file_path, 'r') as f:
        data = f.read()
    assert data.strip() == 'This is a test', 'Failed to verify contents of test file'

    if startup_cmd_file2_path:
        assert os.path.exists(startup_cmd_file2_path), f'Failed to download test file2: {startup_cmd_file2_path}'
        with open(startup_cmd_file2_path, 'r') as f:
            data = f.read()
        assert data.strip() == 'This is a test2', 'Failed to verify contents of test file2'


    assert os.path.exists(shutdown_cmd_file_path), f'Failed to generate shutdown cmd file: {shutdown_cmd_file_path}'
    if shutdown_cmd_file2_path:
        assert os.path.exists(shutdown_cmd_file2_path), f'Failed to generate shutdown cmd file2: {shutdown_cmd_file2_path}'

    assert os.path.exists(model_file_path), f'Failed to download model archive: {model_file_path}'
    assert os.path.exists(model_log_dir), f'Failed to download model training logs: {model_log_dir}'
    assert os.path.exists(upload_file_abspath), f'Failed to upload with abs path: {upload_file_abspath}'

    test_prefix = '.test' if '-test' in model_file_path else ''
    model_name = os.path.basename(model_file_path).replace('.mltk.zip', '').replace('-test', '')
    log_files = [
        f'{model_name}.h5.summary.txt',
        f'{model_name}{test_prefix}.h5',
        f'{model_name}{test_prefix}.tflite',
        f'{model_name}.tflite.summary.txt',
        'train/training-history.json',
        'train/training-history.png',
    ]

    for fn in log_files:
        p = f'{model_log_dir}/{fn}'
        assert os.path.exists(p), f'Missing log file: {p}'
