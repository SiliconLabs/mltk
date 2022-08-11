
import os 
import zipfile

import mltk
from mltk.utils.path import (create_tempdir, remove_directory, fullpath)
from mltk.core.utils import (get_mltk_logger, ArchiveFileNotFoundError)
from .base_mixin import BaseMixin
from ..model_attributes import MltkModelAttributesDecorator


ARCHIVE_EXTENSION = '.mltk.zip'
TEST_ARCHIVE_EXTENSION = '-test.mltk.zip'


@MltkModelAttributesDecorator()
class ArchiveMixin(BaseMixin):
    
    @property 
    def archive_path(self):
        """Return path to model archive file (.mdk.zip)"""
        return f'{os.path.dirname(self.model_specification_path)}/{self.name}{get_archive_extension(test=self.test_mode_enabled)}'

    
    @property
    def h5_archive_path(self):
        """Return path to .h5 model file automatically extracted from model archive file (.mdk.zip)"""
        try:
            ext = '.test.h5' if self.test_mode_enabled else '.h5' 
            return self.get_archive_file(self.name + ext)
        except ArchiveFileNotFoundError:
            # pylint: disable=raise-missing-from
            raise ArchiveFileNotFoundError(
                f'Failed to get .h5 file from model archive: {self.archive_path}\n' \
                'Has the model been trained first?') 


    @property
    def tflite_archive_path(self):
        """Return path to .tflite model file automatically extracted from model's archive file (.mdk.zip)"""
        try:
            ext = '.test.tflite' if self.test_mode_enabled else '.tflite' 
            return self.get_archive_file(self.name + ext)
        except ArchiveFileNotFoundError:
            # pylint: disable=raise-missing-from
            raise ArchiveFileNotFoundError(
                f'Failed to get .tflite file from model archive: {self.archive_path}\n' \
                'Has the model been trained and quantized first?' ) 
    

    def check_archive_file_is_writable(self, throw_exception=False) -> bool:
        """Return if the model archive file is writable"""
        try:
            mode = 'a' if os.path.exists(self.archive_path) else 'w'
            with zipfile.ZipFile(self.archive_path, mode, zipfile.ZIP_DEFLATED):
                pass
            if mode == 'w':
                os.remove(self.archive_path)
            is_writable = True 
        except:
            is_writable = False
        
        if not is_writable:
            msg = f'Model archive file is not writable:\nArchive path:{self.archive_path}\n' + \
            'Ensure that the directory has write permissions and no other programs have this file opened'
            if throw_exception:
                raise Exception(msg)
            get_mltk_logger().warning(msg)

        return is_writable
    

    def get_archive_file(self, name: str, dest_dir:str = None) -> str:
        """Extract a file from the model's archive file"""
        if dest_dir is None:
            dest_dir = create_tempdir(f'models/{self.name}/extracted_archive')
        return extract_file(self.archive_path, name=name, dest_dir=dest_dir)


    def get_archive_dir(self, name: str, dest_dir:str = None) -> str:
        """Extract a directory from the model's archive file"""
        if dest_dir is None:
            dest_dir = create_tempdir(f'models/{self.name}/extracted_archive/{name}')
            remove_directory(dest_dir)
        return extract_dir(self.archive_path, name=name, dest_dir=dest_dir)


    def add_archive_file(self, name):
        """Add given log file in the log directory to the model archive"""

        name = name.replace('\\', '/')

        if name == '__mltk_model_spec__':
            with open(self.model_specification_path, 'r') as fp:
                model_spec_data = fp.read()
            
            tmp_model_spec_path = create_tempdir('scratch') + f'/{self.name}.py'
            with open(tmp_model_spec_path, 'w') as fp:
                fp.write(f"__mltk_version__ = '{mltk.__version__}'\n\n")
                fp.write(model_spec_data)

            file_path = tmp_model_spec_path
            arcname = os.path.basename(file_path)

        elif name.startswith(self.log_dir):
            name = name.replace(f'{self.log_dir}/', '')
            file_path = f'{self.log_dir}/{name}'
            arcname = os.path.relpath(file_path, self.log_dir).replace('\\', '/')
        else:
            file_path = name
            arcname = os.path.basename(file_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File not found: {file_path}')
      
        zip_fp = self._create_archive(exclude=[arcname])
        get_mltk_logger().debug(f'Archiving {file_path} -> {arcname}')
        zip_fp.write(file_path, arcname=arcname)
        zip_fp.close()
 

    def add_archive_dir(self, base_dir, create_new=False, recursive=False):
        """Add given log directory to the model archive"""

        search_dir = f'{self.log_dir}/{base_dir}'

        src_paths = []
        arcnames = []
        for root, _, files in os.walk(search_dir):
            for fn in files:
                src_path = f'{root}/{fn}'.replace('\\', '/')
                arcname = os.path.relpath(src_path, self.log_dir).replace('\\', '/')
                src_paths.append(src_path)
                arcnames.append(arcname)
            
            if not recursive:
                break

        zip_fp = self._create_archive(create_new=create_new, exclude=arcnames)
        for i, src_path in enumerate(src_paths):
            zip_fp.write(src_path, arcname=arcnames[i])
        zip_fp.close()
    

    def _create_archive(self, create_new=False, exclude=None):
        if exclude is None:
            exclude = []
        
        if not create_new and os.path.exists(self.archive_path):
            tmp_dir = create_tempdir(f'tmp_archives/{self.name}')
            remove_directory(tmp_dir)
            with zipfile.ZipFile(self.archive_path, 'r') as zip_fp:
                members = [x for x in zip_fp.namelist() if x not in exclude]
                zip_fp.extractall(tmp_dir, members=members)
            
            zip_fp = zipfile.ZipFile(self.archive_path, 'w', zipfile.ZIP_DEFLATED)
            for root, _, files in os.walk(tmp_dir):
                for fn in files:
                    src_path = f'{root}/{fn}'
                    arcname = os.path.relpath(src_path, tmp_dir).replace('\\', '/')
                    zip_fp.write(src_path, arcname=arcname)

            remove_directory(tmp_dir)
        else:
            zip_fp = zipfile.ZipFile(self.archive_path, 'w', zipfile.ZIP_DEFLATED)

        return zip_fp




def get_archive_extension(test:bool=False) -> str:
    """Return the model archive file extension"""
    return TEST_ARCHIVE_EXTENSION if test else ARCHIVE_EXTENSION



def extract_file(archive_path: str, name: str, dest_dir:str = None) -> str:
    """Extract a file from the give archive"""

    archive_path = fullpath(archive_path)

    if not os.path.exists(archive_path):
        raise ArchiveFileNotFoundError(f'Archive file not found: {archive_path}.\nHas the model been trained first?')
    
    if dest_dir is None:
        model_name = os.path.basename(archive_path[:-len('mltk.zip')])
        dest_dir = create_tempdir(f'models/{model_name}/extracted_files')

    extracted_path = f'{dest_dir}/{name}'
    extracted_dir = os.path.dirname(extracted_path)
    if extracted_dir:
        os.makedirs(extracted_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(archive_path, 'r') as fp:
            try:
                fp.getinfo(name)
            except:
                raise ArchiveFileNotFoundError(f'No file named {name} in {archive_path}') # pylint: disable=raise-missing-from

            get_mltk_logger().debug(f'Extracting {name} -> {dest_dir}')
            with open(extracted_path, 'wb') as dst:
                dst.write(fp.read(name))
    except Exception as e:
        try:
            os.remove(extracted_path)
        except:
            pass 
        raise e

    return extracted_path


def extract_dir(archive_path: str, name: str, dest_dir: str=None) -> str:
    """Extract a directory from the give archive"""

    if not os.path.exists(archive_path):
        raise FileNotFoundError(f'Archive file not found: {archive_path}.\nHas the model been trained first?')
    
    if dest_dir is None:
        model_name = os.path.basename(archive_path[:-len('mltk.zip')])
        dest_dir = create_tempdir(f'models/{model_name}/extracted_files')

    with zipfile.ZipFile(archive_path, 'r') as fp:
        for fn in fp.infolist():
            if not fn.filename.startswith(name):
                continue 
        
            extracted_path = f'{dest_dir}/{fn.filename}'
            os.makedirs(os.path.dirname(extracted_path), exist_ok=True)

            get_mltk_logger().debug(f'Extracting {fn.filename} -> {dest_dir}')
            with open(extracted_path, 'wb') as dst:
                dst.write(fp.read(fn.filename))

    return f'{dest_dir}/{name}'
