from typing import List
from .base_mixin import BaseMixin


from ..model_attributes import MltkModelAttributesDecorator



@MltkModelAttributesDecorator()
class SshMixin(BaseMixin):
    """Provides various properties to the base :py:class:`~MltkModel` used by the ``ssh`` MLTK command.

    .. seealso::
       - `Model Training via SSH <https://siliconlabs.github.io/mltk/docs/guides/model_training_via_ssh.html>`_
       - `Cloud Training with vast.ai <https://siliconlabs.github.io/mltk/mltk/tutorials/cloud_training_with_vast_ai.html>`_
    
    """


    @property
    def ssh_remote_dir(self) -> str:
        """Directory path to remote machine's workspace. This overrides any settings in ~/.mltk/user_settings.yaml
        
        .. seealso:: `Model Training via SSH <https://siliconlabs.github.io/mltk/docs/guides/model_training_via_ssh.html>`_
        """
        return self._attributes.get_value('ssh.remote_dir', default=None)
    @ssh_remote_dir.setter
    def ssh_remote_dir(self, v:str):
        self._attributes['ssh.remote_dir'] = v


    @property
    def ssh_startup_cmds(self) -> List[str]:
        """List of shell commands to execute before invoking MLTK command. This overrides any settings in ~/.mltk/user_settings.yaml
        
        .. seealso:: `Model Training via SSH <https://siliconlabs.github.io/mltk/docs/guides/model_training_via_ssh.html>`_
        """
        return self._attributes.get_value('ssh.startup_cmds', default=None)
    @ssh_startup_cmds.setter
    def ssh_startup_cmds(self, v:List[str]):
        self._attributes['ssh.startup_cmds'] = v


    @property
    def ssh_shutdown_cmds(self) -> List[str]:
        """List of shell commands to execute after invoking MLTK command. This overrides any settings in ~/.mltk/user_settings.yaml
        
        .. seealso:: `Model Training via SSH <https://siliconlabs.github.io/mltk/docs/guides/model_training_via_ssh.html>`_
        """
        return self._attributes.get_value('ssh.shutdown_cmds', default=None)
    @ssh_shutdown_cmds.setter
    def ssh_shutdown_cmds(self, v:List[str]):
        self._attributes['ssh.shutdown_cmds'] = v


    @property
    def ssh_upload_files(self) -> List[str]:
        """List of local files to upload before invoking MLTK command. All paths must be relative to the model specification script.
        This overrides any settings in ~/.mltk/user_settings.yaml
        
        .. seealso:: `Model Training via SSH <https://siliconlabs.github.io/mltk/docs/guides/model_training_via_ssh.html>`_
        """
        return self._attributes.get_value('ssh.upload_files', default=None)
    @ssh_upload_files.setter
    def ssh_upload_files(self, v:List[str]):
        self._attributes['ssh.upload_files'] = v


    @property
    def ssh_download_files(self) -> List[str]:
        """List of remote files to download after invoking MLTK command. All paths must be relative to the model specification script.
        This overrides any settings in ~/.mltk/user_settings.yaml
        
        .. seealso:: `Model Training via SSH <https://siliconlabs.github.io/mltk/docs/guides/model_training_via_ssh.html>`_
        """
        return self._attributes.get_value('ssh.download_files', default=None)
    @ssh_download_files.setter
    def ssh_download_files(self, v:List[str]):
        self._attributes['ssh.download_files'] = v


    @property
    def ssh_environment(self) -> List[str]:
        """List of environment variables to export in the remote machine's shell session before invoking MLTK command.
        This overrides any settings in ~/.mltk/user_settings.yaml
        
        .. seealso:: `Model Training via SSH <https://siliconlabs.github.io/mltk/docs/guides/model_training_via_ssh.html>`_
        """
        return self._attributes.get_value('ssh.environment', default=None)
    @ssh_environment.setter
    def ssh_environment(self, v:List[str]):
        self._attributes['ssh.environment'] = v


    @property
    def ssh_create_venv(self) -> bool:
        """If true then create the MLTK python virtual environment in the specified remote_directory. This overrides the setting in ~/.mltk/user_settings.yaml
        
        .. seealso:: `Model Training via SSH <https://siliconlabs.github.io/mltk/docs/guides/model_training_via_ssh.html>`_
        """
        return self._attributes.get_value('ssh.create_venv', default=None)
    @ssh_create_venv.setter
    def ssh_create_venv(self, v:bool):
        self._attributes['ssh.create_venv'] = v



    def _register_attributes(self):
        self._attributes.register('ssh.remote_dir', dtype=str)
        self._attributes.register('ssh.startup_cmds', dtype=(list,tuple))
        self._attributes.register('ssh.shutdown_cmds', dtype=(list,tuple))
        self._attributes.register('ssh.upload_files', dtype=(list,tuple))
        self._attributes.register('ssh.download_files', dtype=(list,tuple))
        self._attributes.register('ssh.environment', dtype=(dict,list,tuple))
        self._attributes.register('ssh.create_venv', dtype=bool)
